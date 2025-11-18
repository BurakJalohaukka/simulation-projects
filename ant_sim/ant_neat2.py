#!/usr/bin/env python3
import math
import random
import numpy as np
import pygame
import neat
import pickle

# ============================================================
# CONFIG
# ============================================================
WIDTH, HEIGHT = 600, 600
ANT_SIZE = 5
FOOD_SIZE = 6

MAX_SPEED = 2.2
TURN_SPEED = 0.22

SIMULATION_STEPS = 500  # increase if needed

# ============================================================
# ENVIRONMENT
# ============================================================
class Ant:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0

        self.prev_dist = None
        self.time_alive = 0
        self.fitness = 0

    def move(self, turn, accel):
        # clamp inputs
        turn = max(-1, min(1, turn))
        accel = max(-1, min(1, accel))

        # rotation
        self.angle += turn * TURN_SPEED

        # forward/backward
        self.speed += accel * 0.1
        self.speed = max(0, min(MAX_SPEED, self.speed))

        # movement
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # wall collision
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

        return hit_wall

    def get_observation(self, food):
        dx = food.x - self.x
        dy = food.y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        angle_to_food = math.atan2(dy, dx)
        angle_diff = angle_to_food - self.angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        # normalize
        obs = [
            math.cos(self.angle),
            math.sin(self.angle),
            self.speed / MAX_SPEED,
            dx / WIDTH,
            dy / HEIGHT,
            dist / math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT),
            math.cos(angle_diff),
            math.sin(angle_diff),
        ]

        return obs, dist

class Food:
    def __init__(self):
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)

# ============================================================
# SIMULATION LOOP
# ============================================================
def eval_genomes(genomes, config):

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    clock = pygame.time.Clock()

    nets = []
    ants = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        nets.append(net)
        ants.append(Ant())
        ge.append(genome)

    food = Food()

    for step in range(SIMULATION_STEPS):

        # handle close window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for i, ant in enumerate(ants):
            obs, dist = ant.get_observation(food)
            outputs = nets[i].activate(obs)

            turn = outputs[0]
            accel = outputs[1]

            hit_wall = ant.move(turn, accel)

            # reward logic
            if ant.prev_dist is None:
                ant.prev_dist = dist

            # moving closer to the food
            if dist < ant.prev_dist:
                ge[i].fitness += 0.5
            else:
                ge[i].fitness -= 0.1

            # punish wall hugging
            if hit_wall:
                ge[i].fitness -= 0.3

            # found food
            if dist < 10:
                ge[i].fitness += 50
                food = Food()  # respawn food

            ant.prev_dist = dist

        # DRAW
        screen.fill((30, 30, 30))

        pygame.draw.circle(screen, (0, 255, 0), (int(food.x), int(food.y)), FOOD_SIZE)

        for ant in ants:
            pygame.draw.circle(screen, (200, 200, 255), (int(ant.x), int(ant.y)), ANT_SIZE)

        pygame.display.flip()
        clock.tick(60)


# ============================================================
# MAIN
# ============================================================
def run():
    config_path = "config-neat.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(eval_genomes, 40)

    with open("winner_ant.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved to 'winner_ant.pkl'.")
    

    print("Winner:", winner)


if __name__ == "__main__":
    run()

