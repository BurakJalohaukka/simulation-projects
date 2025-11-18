#!/usr/bin/env python3
"""
ant_neat_pheromone.py
Continuous ant simulation with pheromone trails and multiple-food foraging.
One ant per genome (per-evaluation). Each genome is evaluated in its own environment
(with its own pheromone grid) so training stays deterministic.

Usage:
    python3 ant_neat_pheromone.py
Requires:
    pip install pygame neat-python numpy
"""

import math
import random
import pickle
import os
import sys
import neat
import pygame
import numpy as np

# -------------------
# CONFIG / HYPERPARAMS
# -------------------
WIDTH, HEIGHT = 600, 600
ANT_SIZE = 5
FOOD_SIZE = 6

MAX_SPEED = 2.2
TURN_SPEED = 0.22

SIMULATION_STEPS = 600   # steps per episode
EVAL_EPISODES = 3        # average fitness over these episodes

PHER_GRID_SIZE = 60      # pheromone grid resolution (PHER_GRID_SIZE x PHER_GRID_SIZE)
PHER_EVAP_RATE = 0.994   # multiply pheromone map by this each step
PHER_DEPOSIT = 1.0       # deposit per step when not carrying
PHER_DEPOSIT_CARRY = 2.5 # deposit per step when carrying
PHER_SENSOR_DIST = 25.0  # distance in pixels to sample pheromone sensors
PHER_SENSOR_ANGLE = math.radians(30)  # left/right sensor offset

FOOD_COUNT = 5
NEST_POS = (WIDTH // 2, HEIGHT // 2)
PICKUP_RADIUS = 10
NEST_RADIUS = 18

# Reward scalars
PROGRESS_SCALE = 1.5
PICKUP_REWARD = 40.0
DELIVER_REWARD = 150.0
WALL_PENALTY = -0.6
REVISIT_PENALTY = -0.35

# Rendering
FPS = 60
VISUALIZE_AFTER_TRAIN = True

# -------------------
# UTIL
# -------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------
# ENVIRONMENT CLASSES
# -------------------
class PheromoneMap:
    """Simple 2D pheromone grid with evaporation."""
    def __init__(self, width_pixels, height_pixels, grid_size=PHER_GRID_SIZE):
        self.grid_size = grid_size
        self.cell_w = width_pixels / grid_size
        self.cell_h = height_pixels / grid_size
        self.map = np.zeros((grid_size, grid_size), dtype=np.float32)

    def evaporate(self):
        # exponential decay
        self.map *= PHER_EVAP_RATE

    def deposit_at_pixel(self, px, py, amount):
        gx = int(px / self.cell_w)
        gy = int(py / self.cell_h)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.map[gy, gx] += amount

    def sample_pixel(self, px, py):
        gx = int(px / self.cell_w)
        gy = int(py / self.cell_h)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            # normalize sample to [0,1] by dividing by a softcap
            return float(self.map[gy, gx] / (1.0 + self.map[gy, gx]))
        return 0.0

    def deposit_line(self, px, py, amount):
        # convenience wrapper (same as deposit_at_pixel)
        self.deposit_at_pixel(px, py, amount)

class Food:
    def __init__(self):
        self.respawn()

    def respawn(self):
        # ensure not too close to nest
        while True:
            self.x = random.randint(30, WIDTH - 30)
            self.y = random.randint(30, HEIGHT - 30)
            if euclidean((self.x, self.y), NEST_POS) > 60:
                break

class Ant:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.prev_target_dist = None
        self.carrying = False
        # visitation grid coarse (for revisit penalty)
        self.visit_counts = {}

    def move(self, turn, accel):
        # clamp inputs
        turn = clamp(turn, -1.0, 1.0)
        accel = clamp(accel, -1.0, 1.0)

        # rotation and acceleration
        self.angle += turn * TURN_SPEED
        self.speed += accel * 0.1
        self.speed = max(0.0, min(MAX_SPEED, self.speed))

        # integrate
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # wall collisions (keep inside)
        hit_wall = False
        if self.x < 0:
            self.x = 0
            hit_wall = True
        if self.x > WIDTH:
            self.x = WIDTH
            hit_wall = True
        if self.y < 0:
            self.y = 0
            hit_wall = True
        if self.y > HEIGHT:
            self.y = HEIGHT
            hit_wall = True

        # update visits
        gx = int(self.x // 20)  # coarse cell for revisit penalty
        gy = int(self.y // 20)
        key = (gx, gy)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

        return hit_wall

    def get_target_vector(self, foods):
        """
        If carrying -> target is nest.
        Else -> target is nearest food (euclidean).
        Returns (tx, ty, dist)
        """
        if self.carrying:
            tx, ty = NEST_POS
        else:
            # nearest food
            nearest = min(foods, key=lambda f: euclidean((f.x, f.y), (self.x, self.y)))
            tx, ty = nearest.x, nearest.y
        dist = euclidean((tx, ty), (self.x, self.y))
        return (tx, ty, dist)

    def get_observation(self, foods, pher_map):
        # target depends on carrying
        tx, ty, dist = self.get_target_vector(foods)

        dx = tx - self.x
        dy = ty - self.y
        # angular difference to target
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - self.angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        # normalized values
        obs = []
        obs.append(math.cos(self.angle))
        obs.append(math.sin(self.angle))
        obs.append(self.speed / MAX_SPEED)
        obs.append(dx / float(WIDTH))
        obs.append(dy / float(HEIGHT))
        # normalize distance by diagonal
        obs.append(dist / math.hypot(WIDTH, HEIGHT))
        obs.append(math.cos(angle_diff))
        obs.append(math.sin(angle_diff))

        # Pheromone sensors: ahead, left, right
        # sample points in world coordinates
        def sample_at(angle_offset):
            a = self.angle + angle_offset
            sx = self.x + math.cos(a) * PHER_SENSOR_DIST
            sy = self.y + math.sin(a) * PHER_SENSOR_DIST
            return pher_map.sample_pixel(sx, sy)

        pher_ahead = sample_at(0.0)
        pher_left = sample_at(PHER_SENSOR_ANGLE)
        pher_right = sample_at(-PHER_SENSOR_ANGLE)

        obs.append(pher_ahead)
        obs.append(pher_left)
        obs.append(pher_right)

        obs.append(1.0 if self.carrying else 0.0)

        return obs, dist

# -------------------
# EVALUATION
# -------------------
def evaluate_genome_once(genome, config, render=False, screen=None, clock=None):
    """
    Run one episode for a genome in its own env (pheromone map reset).
    Returns total episode fitness.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ant = Ant()
    foods = [Food() for _ in range(FOOD_COUNT)]
    pher = PheromoneMap(WIDTH, HEIGHT, grid_size=PHER_GRID_SIZE)

    total_reward = 0.0
    ant.prev_target_dist = ant.get_target_vector(foods)[2]

    # For rendering
    if render and (screen is None or clock is None):
        raise ValueError("screen and clock required for render=True")

    for step in range(SIMULATION_STEPS):
        # render if needed
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # get observation
        obs, dist = ant.get_observation(foods, pher)
        outputs = net.activate(obs)
        # outputs: [turn, accel] — they are raw floats, in range approx [-..+..]
        turn = outputs[0]
        accel = outputs[1]

        # move ant
        hit_wall = ant.move(turn, accel)

        # deposit pheromone at ant position
        deposit_amt = PHER_DEPOSIT_CARRY if ant.carrying else PHER_DEPOSIT
        pher.deposit_at_pixel(ant.x, ant.y, deposit_amt)

        # evaporation (per step)
        pher.evaporate()

        # check pickup
        if not ant.carrying:
            # find any food within pickup radius
            for f in foods:
                if euclidean((f.x, f.y), (ant.x, ant.y)) <= PICKUP_RADIUS:
                    ant.carrying = True
                    total_reward += PICKUP_REWARD
                    # mark this food as taken — respawn it elsewhere
                    f.respawn()
                    break

        else:
            # carrying -> check if at nest
            if euclidean((ant.x, ant.y), NEST_POS) <= NEST_RADIUS:
                ant.carrying = False
                total_reward += DELIVER_REWARD
                # optionally respawn a random food to keep task continuous
                # spawn one new food
                # choose random food to respawn
                random.choice(foods).respawn()

        # progress toward current target (food when not carrying, nest when carrying)
        new_target_dist = ant.get_target_vector(foods)[2]
        progress = (ant.prev_target_dist - new_target_dist)
        total_reward += progress * PROGRESS_SCALE
        ant.prev_target_dist = new_target_dist

        # penalties
        if hit_wall:
            total_reward += WALL_PENALTY

        # revisit penalty (coarse)
        gx = int(ant.x // 20)
        gy = int(ant.y // 20)
        visits = ant.visit_counts.get((gx, gy), 0)
        if visits > 4:
            total_reward += REVISIT_PENALTY

        # small time penalty to encourage efficiency
        total_reward -= 0.001

        # optional rendering
        if render:
            # draw background
            screen.fill((20, 20, 20))
            # draw pheromones as faint purple overlay
            # sample a few points to render grid
            cell_w = WIDTH / pher.grid_size
            cell_h = HEIGHT / pher.grid_size
            for gy_i in range(pher.grid_size):
                for gx_i in range(pher.grid_size):
                    v = pher.map[gy_i, gx_i]
                    if v > 0.001:
                        alpha = clamp(v / 4.0, 0.01, 0.6)
                        color = (int(120 * alpha), int(0 * alpha), int(160 * alpha))
                        rect = pygame.Rect(int(gx_i * cell_w), int(gy_i * cell_h),
                                           int(cell_w) + 1, int(cell_h) + 1)
                        s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                        s.fill((120, 0, 160, int(255 * alpha)))
                        screen.blit(s, rect.topleft)

            # draw foods
            for f in foods:
                pygame.draw.circle(screen, (0, 255, 0), (int(f.x), int(f.y)), FOOD_SIZE)

            # draw nest
            pygame.draw.circle(screen, (200, 200, 80), NEST_POS, NEST_RADIUS, 2)

            # draw ant
            ax, ay = int(ant.x), int(ant.y)
            pygame.draw.circle(screen, (200, 200, 255), (ax, ay), ANT_SIZE)
            # direction line
            dx = int(math.cos(ant.angle) * (ANT_SIZE + 6))
            dy = int(math.sin(ant.angle) * (ANT_SIZE + 6))
            pygame.draw.line(screen, (255, 100, 100), (ax, ay), (ax + dx, ay + dy), 2)

            pygame.display.flip()
            clock.tick(FPS)

    return total_reward

def eval_genomes(genomes, config):
    """
    NEAT eval function: evaluate each genome across EVAL_EPISODES episodes and set genome.fitness
    """
    # Prepare rendering only for visualization of best later (we don't render during training)
    for gid, genome in genomes:
        genome.fitness = 0.0

    for gid, genome in genomes:
        fitnesses = []
        # Average multiple episodes to reduce randomness
        for ep in range(EVAL_EPISODES):
            # Do not render during training (slow). Pass render=False.
            f = evaluate_genome_once(genome, config, render=False, screen=None, clock=None)
            fitnesses.append(f)
        genome.fitness = float(sum(fitnesses) / len(fitnesses))

# -------------------
# MAIN: NEAT RUN
# -------------------
def run(config_file):
    pygame.init()
    screen = None
    clock = None
    if VISUALIZE_AFTER_TRAIN:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    generation_limit = 200  # increase for more training
    winner = p.run(eval_genomes, generation_limit)

    # save winner genome
    with open("winner_ant.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Saved winner -> winner_ant.pkl")

    # visualize winner
    if VISUALIZE_AFTER_TRAIN:
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        # run one long episode with rendering and the winner network
        ant = Ant()
        foods = [Food() for _ in range(FOOD_COUNT)]
        pher = PheromoneMap(WIDTH, HEIGHT, grid_size=PHER_GRID_SIZE)
        ant.prev_target_dist = ant.get_target_vector(foods)[2]

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # one step
            obs, _ = ant.get_observation(foods, pher)
            outputs = net.activate(obs)
            turn, accel = outputs[0], outputs[1]
            hit_wall = ant.move(turn, accel)

            deposit_amt = PHER_DEPOSIT_CARRY if ant.carrying else PHER_DEPOSIT
            pher.deposit_at_pixel(ant.x, ant.y, deposit_amt)
            pher.evaporate()

            # pickups / deliveries
            if not ant.carrying:
                for f in foods:
                    if euclidean((f.x, f.y), (ant.x, ant.y)) <= PICKUP_RADIUS:
                        ant.carrying = True
                        f.respawn()
                        break
            else:
                if euclidean((ant.x, ant.y), NEST_POS) <= NEST_RADIUS:
                    ant.carrying = False
                    random.choice(foods).respawn()

            # redraw
            screen.fill((20, 20, 20))
            cell_w = WIDTH / pher.grid_size
            cell_h = HEIGHT / pher.grid_size
            for gy_i in range(pher.grid_size):
                for gx_i in range(pher.grid_size):
                    v = pher.map[gy_i, gx_i]
                    if v > 0.001:
                        alpha = clamp(v / 4.0, 0.01, 0.6)
                        s = pygame.Surface((int(cell_w)+1, int(cell_h)+1), pygame.SRCALPHA)
                        s.fill((120, 0, 160, int(255 * alpha)))
                        screen.blit(s, (int(gx_i*cell_w), int(gy_i*cell_h)))

            for f in foods:
                pygame.draw.circle(screen, (0, 255, 0), (int(f.x), int(f.y)), FOOD_SIZE)
            pygame.draw.circle(screen, (200, 200, 80), NEST_POS, NEST_RADIUS, 2)

            ax, ay = int(ant.x), int(ant.y)
            pygame.draw.circle(screen, (200, 200, 255), (ax, ay), ANT_SIZE)
            dx = int(math.cos(ant.angle) * (ANT_SIZE + 6))
            dy = int(math.sin(ant.angle) * (ANT_SIZE + 6))
            pygame.draw.line(screen, (255, 100, 100), (ax, ay), (ax + dx, ay + dy), 2)

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat2.txt")
    if not os.path.exists(config_path):
        print("Missing 'config-neat.txt' in the same folder.")
        sys.exit(1)
    run(config_path)

