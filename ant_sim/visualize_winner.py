import math
import random
import pickle
import pygame
import neat
import numpy as np

# ============================================================
# CONFIG (must match training)
# ============================================================
WIDTH, HEIGHT = 600, 600
ANT_SIZE = 5
FOOD_SIZE = 6
MAX_SPEED = 2.2
TURN_SPEED = 0.22

SIMULATION_STEPS = 1500   # replay can be longer


# ============================================================
# ANT & FOOD (same as training)
# ============================================================
class Ant:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0

    def move(self, turn, accel):
        turn = max(-1, min(1, turn))
        accel = max(-1, min(1, accel))

        self.angle += turn * TURN_SPEED

        self.speed += accel * 0.1
        self.speed = max(0, min(MAX_SPEED, self.speed))

        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        if self.x < 0: self.x = 0
        if self.x > WIDTH: self.x = WIDTH
        if self.y < 0: self.y = 0
        if self.y > HEIGHT: self.y = HEIGHT

    def get_observation(self, food):
        dx = food.x - self.x
        dy = food.y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        angle_to_food = math.atan2(dy, dx)
        angle_diff = angle_to_food - self.angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

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
# REPLAY
# ============================================================
def replay():
    # load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-neat.txt"
    )

    # load winner genome
    with open("winner_ant.pkl", "rb") as f:
        genome = pickle.load(f)

    # build winner network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    ant = Ant()
    food = Food()

    for step in range(SIMULATION_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        obs, dist = ant.get_observation(food)
        outputs = net.activate(obs)

        turn, accel = outputs[0], outputs[1]
        ant.move(turn, accel)

        # respawn food if touched
        if dist < 10:
            food = Food()

        # draw
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (0, 255, 0), (int(food.x), int(food.y)), FOOD_SIZE)
        pygame.draw.circle(screen, (200, 200, 255), (int(ant.x), int(ant.y)), ANT_SIZE)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    replay()
