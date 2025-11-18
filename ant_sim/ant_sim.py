import pygame
import random

# -- CONFIG --
GRID_SIZE = 40
CELL_SIZE = 15

WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 10 # game speed

# Colors
ANT_COLOR = (200, 50, 50)
FOOD_COLOR = (50, 200, 50)
BG_COLOR = (30, 30, 30)

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Food Simulation")
clock = pygame.time.Clock()

# Ant starting position
ant_x = GRID_SIZE // 2
ant_y = GRID_SIZE // 2

# Random food spawn
food_x = random.randint(0, GRID_SIZE -1)
food_y = random.randint(0, GRID_SIZE -1)

running = True

while running:
    clock.tick(FPS)

    # --- Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Movement (Random) ---
    direction = random.choice(["up", "down", "left", "right"])

    if direction == "up":
        ant_y = max(0, ant_y - 1)
    elif direction == "down":
        ant_y = min(GRID_SIZE - 1, ant_y + 1)
    elif direction == "left":
        ant_x = max(0, ant_x - 1)
    elif direction == "right":
        ant_x = min(GRID_SIZE - 1, ant_x + 1)

    # --  Check for food Collision ---
    if ant_x == food_x and ant_y == food_y:
        print("Food found! Respawning...")
        food_x = random.randint(0, GRID_SIZE - 1)
        food_y = random.randint(0, GRID_SIZE - 1)

    # --- DRAW ---
    win.fill(BG_COLOR)

    # --- Draw food
    pygame.draw.rect(win, FOOD_COLOR, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw ant
    pygame.draw.rect(win, ANT_COLOR, (ant_x * CELL_SIZE, ant_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()

pygame.quit()

