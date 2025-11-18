"""
ant_neat.py
NEAT-controlled ant that learns to find food on a grid using neat-python and pygame

Usage:
    - Ensure 'config-neat.txt' is in the same folder.
    - Run: python3 ant_neat.py
    - After training, press ESC or close the window to stop.
"""

import math
import random
import os
import sys
import neat
import pygame
import pickle

# -- SIM CONFIG ---
GRID_SIZE = 10 # width/height in cells (smaller for faster training)
CELL_SIZE = 25
SCREEN_W = GRID_SIZE * CELL_SIZE
SCREEN_H = GRID_SIZE * CELL_SIZE

FPS = 10 # render FPS
MAX_STEPS = 200 # ,ax steps per genome episode
EVAL_EPISODES = 3 # run each genome this many episodes for fitness stability

# Colors
BG = (30, 30, 30)
ANT_COLOR = (220, 60, 60)
FOOD_COLOR = (60, 200, 60)

# Remder settings
RENDER_BEST = True # set to True to render best genome during training (slows training)
VISUALIZE_AFTER_TRAIN = True # Load best genome and visualize at the end 

# ENVIRONMENT / UTILS 
def clamp(v, a, b):
    return max(a, min(b, v))

def manhattan(ax, ay, bx, by):
    return abs(ax -bx) + abs(ay - by)

def euclidean(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class AntEnv:
    """
    Clean and consistent grid-world environment for NEAT.
    Observations (8 values):
        0: dx_norm       (food_x - ant_x) / grid
        1: dy_norm       (food_y - ant_y) / grid
        2: nx            normalized ant_x
        3: ny            normalized ant_y
        4: wall_up
        5: wall_down
        6: wall_left
        7: wall_right
    """

    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Start agent in the center
        self.ant_x = self.grid_size // 2
        self.ant_y = self.grid_size // 2

        # Place food randomly (not on ant)
        while True:
            self.food_x = random.randint(0, self.grid_size - 1)
            self.food_y = random.randint(0, self.grid_size - 1)
            if (self.food_x, self.food_y) != (self.ant_x, self.ant_y):
                break

        self.steps = 0
        self.done = False

        # distance for shaping
        self.prev_dist = euclidean(self.ant_x, self.ant_y,
                                   self.food_x, self.food_y)

        # track visited tiles for exploration reward
        self.visits = [[0 for _ in range(self.grid_size)]
                           for _ in range(self.grid_size)]
        self.visits[self.ant_y][self.ant_x] = 1

        return self._get_obs()

    # ----------------------------------------------------------
    # 🔹 FIXED OBSERVATION FUNCTION (8 inputs exactly)
    # ----------------------------------------------------------
    def _get_obs(self):
        dx = (self.food_x - self.ant_x) / float(self.grid_size)
        dy = (self.food_y - self.ant_y) / float(self.grid_size)

        nx = self.ant_x / float(self.grid_size - 1)
        ny = self.ant_y / float(self.grid_size - 1)

        wall_up    = 1.0 if self.ant_y == 0 else 0.0
        wall_down  = 1.0 if self.ant_y == self.grid_size - 1 else 0.0
        wall_left  = 1.0 if self.ant_x == 0 else 0.0
        wall_right = 1.0 if self.ant_x == self.grid_size - 1 else 0.0

        return [dx, dy, nx, ny,
                wall_up, wall_down, wall_left, wall_right]


    # ----------------------------------------------------------
    # FIXED STEP: small, well-scaled env reward (no big novelty here)
    # ----------------------------------------------------------
    def step(self, action):
        """
        Executes `action` and returns (obs, reward, done, info).
        Reward = distance progress * 1.5  (primary)
               + wall penalty if hit
               + big food bonus on success
        Note: We DO NOT give a +1 exploration reward here (that was too large).
        eval_genomes handles the small exploration bonus instead.
        """
        old_x, old_y = self.ant_x, self.ant_y

        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0:
            self.ant_y = max(0, self.ant_y - 1)
        elif action == 1:
            self.ant_y = min(self.grid_size - 1, self.ant_y + 1)
        elif action == 2:
            self.ant_x = max(0, self.ant_x - 1)
        elif action == 3:
            self.ant_x = min(self.grid_size - 1, self.ant_x + 1)

        hit_wall = (self.ant_x == old_x and self.ant_y == old_y)

        # Distance shaping (primary scalar reward)
        new_dist = euclidean(self.ant_x, self.ant_y,
                             self.food_x, self.food_y)
        reward = (self.prev_dist - new_dist) * 2.0   # make progress signal stronger
        self.prev_dist = new_dist

        # Wall penalty (slightly stronger)
        if hit_wall:
            reward -= 0.8

        # Food success
        done = (self.ant_x == self.food_x and self.ant_y == self.food_y)
        if done:
            reward += 100.0   # larger terminal bonus — makes solving worthwhile

        # record visit count for external logic
        self.visits[self.ant_y][self.ant_x] += 1

        self.steps += 1
        return self._get_obs(), reward, done, {"hit_wall": hit_wall}


# --- PYGAME RENDER ---
def render_pygame(screen, env):
    screen.fill(BG)

    # draw grid optional (comment out for speed)
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    # Draw food
    fx = env.food_x * CELL_SIZE
    fy = env.food_y * CELL_SIZE
    pygame.draw.rect(screen, FOOD_COLOR, (fx, fy, CELL_SIZE, CELL_SIZE))

    # draw ant
    ax = env.ant_x * CELL_SIZE
    ay = env.ant_y * CELL_SIZE
    pygame.draw.rect(screen, ANT_COLOR, (ax, ay, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()

# --- NEAT Evalutation ---
def evaluate_genome_once(genome, config, render=False, screen=None, clock=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = AntEnv()
    obs = env.reset()
    total_reward = 0.0

    if render and screen is None:
        raise ValueError("screen required for render=True")

    while True:
        # network forward
        outputs = net.activate(obs)
        # choose action
        # support networks that output arbitrary floats; choose argmax
        action = int(max(range(len(outputs)), key=lambda i: outputs[i]))

        obs, reward, done, _= env.step(action)
        total_reward += reward

        if render:
            render_pygame(screen, env)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            clock.ticks(FPS)

        if done:
            break

    return total_reward, env.steps




def eval_genomes(genomes, config):
    opposites = {0:1, 1:0, 2:3, 3:2}

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = AntEnv()
        genome.fitness = 0.0

        # Initial distance to food
        prev_dist = math.dist((env.ant_x, env.ant_y),
                              (env.food_x, env.food_y))

        prev_action_opposite = None
        # track visited positions using env.visits (env updates it)
        # visited_bonus will be applied only once per new tile (small)
        visited_set = set()
        visited_set.add((env.ant_x, env.ant_y))

        for step in range(MAX_STEPS):
            obs = env._get_obs()
            outputs = net.activate(obs)
            action = outputs.index(max(outputs))

            obs, reward, done, info = env.step(action)
            hit_wall = info.get("hit_wall", False)

            # ---------------------------------------
            # 1) Add env reward (distance progress, wall penalty, food bonus)
            # env.step already returned that number (scaled).
            # Use it directly but also keep a small multiplier if needed.
            # ---------------------------------------
            genome.fitness += reward  # primary signal

            # ---------------------------------------
            # 2) Small novelty reward for *new* tile (IMPORTANT: small!)
            #    — gives curiosity but not dominating the objective
            # ---------------------------------------
            pos = (env.ant_x, env.ant_y)
            if pos not in visited_set:
                genome.fitness += 0.15   # small exploration bonus
                visited_set.add(pos)

            # ---------------------------------------
            # 3) Penalize revisiting same tile repeatedly (discourage loops)
            # ---------------------------------------
            visits_here = env.visits[env.ant_y][env.ant_x]
            if visits_here > 3:
                genome.fitness -= 0.3  # stronger penalty for repeating

            # ---------------------------------------
            # 4) Anti-oscillation: punish immediate reversal
            # ---------------------------------------
            if prev_action_opposite is not None and action == prev_action_opposite:
                genome.fitness -= 0.7
            prev_action_opposite = opposites[action]

            # ---------------------------------------
            # 5) Encourage staying away from walls (small)
            # ---------------------------------------
            dist_to_wall = min(env.ant_x, env.ant_y,
                               env.grid_size - env.ant_x - 1,
                               env.grid_size - env.ant_y - 1)
            if dist_to_wall <= 1:
                genome.fitness -= 0.2

            # ---------------------------------------
            # 6) Terminal reward already applied via env.step; break if solved
            # ---------------------------------------
            if done:
                # env already added +100; you can add tiny extra if desired
                genome.fitness += 0.0
                break

# -- MAIN: Setup NEAT, run evolution ---
def run(config_file):
    # Initialize pygame only if visulation during/after training
    pygame.init()
    screen = None
    clock = None
    if RENDER_BEST or VISUALIZE_AFTER_TRAIN:
        # create a screen but we won't render every genome's run (too slow)
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("NEAT Ant Simulation")
        clock = pygame.time.Clock()
    config = neat.Config (neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # create the population add reporters
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # run NEAT
    generation_limit = 300 # Can be increased (100+) for stronger performance
    winner = p.run(eval_genomes, generation_limit)

    # Save winner
    with open("winner_ant.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved to 'winner_ant.pkl'.")

    # Visualize the winner in pygame window (optional)
    if VISUALIZE_AFTER_TRAIN:
        print("Visualizing best genome. Close window or press ESC to quit.")
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        env = AntEnv()
        obs = env.reset()
        running = True
        while running:
            outputs = net.activate(obs)
            action = int(max(range(len(outputs)), key=lambda i: outputs[i]))
            obs, _, done, _ = env.step(action)

            render_pygame(screen, env)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            clock.tick(FPS)
            if done:
                # respawn food after short pause
                pygame.time.delay(700)
                obs = env.reset()
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat.txt")
    if not os.path.exists(config_path):
        print("ERROR: 'config-neat.txt' not found in the current directory.")
        sys.exit(1)
    run(config_path)

