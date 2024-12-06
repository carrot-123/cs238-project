import pygame
import time
import random

# press game to replay

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Block size for the snake and food
BLOCK_SIZE = 20

# FPS controller
FPS = 15

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Clock
clock = pygame.time.Clock()

# Font for displaying score
font = pygame.font.SysFont("bahnschrift", 25)
game_over_font = pygame.font.SysFont("bahnschrift", 40)


def show_score(score):
    """Displays the current score on the screen."""
    value = font.render(f"Score: {score}", True, WHITE)
    screen.blit(value, [10, 10])

def game_over_message():
    """Displays the game over message."""
    message = game_over_font.render("Game Over! Press any key to play again.", True, RED)
    screen.blit(message, [WIDTH // 6, HEIGHT // 2])

def game_loop():
    # Snake starting position
    snake_pos = [200, 200]
    snake_body = [[200, 200]] #, [80, 60], [60, 60]

    # Direction variables
    direction = 'RIGHT'
    change_to = direction

    # Initial score
    score = 0

    # Food position
    food_pos = [
        random.randrange(0, (WIDTH // BLOCK_SIZE)) * BLOCK_SIZE,
        random.randrange(0, (HEIGHT // BLOCK_SIZE)) * BLOCK_SIZE
    ]
    food_spawn = True

    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    change_to = 'UP'
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    change_to = 'DOWN'
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    change_to = 'LEFT'
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    change_to = 'RIGHT'

        # Update direction
        direction = change_to

        # Move the snake
        if direction == 'UP':
            snake_pos[1] -= BLOCK_SIZE
        if direction == 'DOWN':
            snake_pos[1] += BLOCK_SIZE
        if direction == 'LEFT':
            snake_pos[0] -= BLOCK_SIZE
        if direction == 'RIGHT':
            snake_pos[0] += BLOCK_SIZE

        # Snake growing mechanism: Add new head
        snake_body.insert(0, list(snake_pos))

        # Check if the snake eats the food
        if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
            score += 10
            food_spawn = False
        else:
            # Remove the last segment if no food eaten
            snake_body.pop()

        # Respawn food
        if not food_spawn:
            food_pos = [
                random.randrange(0, (WIDTH // BLOCK_SIZE)) * BLOCK_SIZE,
                random.randrange(0, (HEIGHT // BLOCK_SIZE)) * BLOCK_SIZE
            ]
        food_spawn = True

        # Game over conditions: Wall collision or self-collision
        if (
            snake_pos[0] < 0 or snake_pos[0] >= WIDTH or
            snake_pos[1] < 0 or snake_pos[1] >= HEIGHT
        ):
            break
        for block in snake_body[1:]:
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                break

        # Update the screen
        screen.fill(BLACK)

        # Draw the snake
        for block in snake_body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))

        # Draw the food
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], BLOCK_SIZE, BLOCK_SIZE))

        # Display the score
        show_score(score)

        # Refresh the screen
        pygame.display.update()

        # Control the game speed
        clock.tick(FPS)

    while True:
        screen.fill(BLACK)
        game_over_message()
        pygame.display.update()

        # Wait for any key press to restart the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
              return  # Restart the game loop

if __name__ == "__main__":
    while True:
        game_loop()  # Restart the game when the loop ends