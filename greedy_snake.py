
import pygame
import time
import random
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np



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
BLUE = (0, 0, 255)

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
    # load in model
    
    time.sleep(5)
    # add manual delay here later

    pyautogui.PAUSE = 0.25

    scores = []
    steps = []
    
    # play 100 games, track the average reward and the average time elapsed
    for episode in range(100):
        # Snake starting position
        snake_pos = [20, 20]
        snake_body = [[20, 20]]#[[80, 20], [60, 20], [40, 20]]

        # Direction variables
        direction = 'RIGHT'
        change_to = direction

        # Initial score
        score = 0

        # Intial steps
        step = 0

        # Food position
        while True:
            # Generate random coordinates within the grid
            food_pos = [
                random.randrange(0, (WIDTH // BLOCK_SIZE)) * BLOCK_SIZE,
                random.randrange(0, (HEIGHT // BLOCK_SIZE)) * BLOCK_SIZE
            ]

            # Ensure the food doesn't spawn on the snake's body
            if food_pos not in snake_body:
                break
        food_spawn = True
       
        done = False
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        #state = get_state(snake_pos, snake_body, direction, food_pos)
        while not done:
            action = greedy_move(snake_pos, snake_body, food_pos)
            #action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()
            #print(action)
            button = actions[action]
            pyautogui.press(button)
            step += 1

            # Event handling
            for event in pygame.event.get():
                
                # use pyautogui here to press
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
                while True:
                    # Generate random coordinates within the grid
                    food_pos = [
                        random.randrange(0, (WIDTH // BLOCK_SIZE)) * BLOCK_SIZE,
                        random.randrange(0, (HEIGHT // BLOCK_SIZE)) * BLOCK_SIZE
                    ]

                    # Ensure the food doesn't spawn on the snake's body
                    if food_pos not in snake_body:
                        break
            food_spawn = True

            #state = get_state(snake_pos, snake_body, direction, food_pos)

            # Game over conditions: Wall collision or self-collision
            if (
                snake_pos[0] < 0 or snake_pos[0] >= WIDTH or
                snake_pos[1] < 0 or snake_pos[1] >= HEIGHT
            ):
                done = True
            for block in snake_body[1:]:
                if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                    done = True

            # Update the screen
            screen.fill(BLACK)

            # Draw the snake
            for block in snake_body:
                pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, BLUE, pygame.Rect(snake_pos[0], snake_pos[1], BLOCK_SIZE, BLOCK_SIZE))

            # Draw the food
            pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], BLOCK_SIZE, BLOCK_SIZE))

            # Display the score
            show_score(score)

            # Refresh the screen
            pygame.display.update()

            # Control the game speed
            #clock.tick(FPS)
        scores.append(score)
        steps.append(step)
    print("mean score: " + str(np.mean(scores)))
    print("median score: " + str(np.median(scores)))
    print("min score: " + str(np.min(scores)))
    print("max score: " + str(np.max(scores)))
    print("----------")
    print("mean score: " + str(np.mean(scores)))
    print("median score: " + str(np.median(scores)))
    print("min score: " + str(np.min(scores)))
    print("max score: " + str(np.max(scores)))

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




# Helper function to calculate Manhattan distance
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])



# Greedy algorithm to choose the next move
def greedy_move(snake_pos, snake_body, food):
  UP = (0, -1)
  DOWN = (0, 1)
  LEFT = (-1, 0)
  RIGHT = (1, 0)

  DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
  
  head = snake_pos
  best_move = 0
  min_distance = float('inf')

  for i in range(len(DIRECTIONS)):
      direction = DIRECTIONS[i]
      # Calculate new head position
      new_head = (head[0] + direction[0], head[1] + direction[1])
      
      # Check if the move is valid
      #[snake_x, snake_y - BLOCK_SIZE], snake_body)
      if is_obstacle(new_head, snake_body) == 0:
          # Calculate distance to the food
          distance = manhattan_distance(new_head, food)
          if distance < min_distance:
              min_distance = distance
              # best_move = new_head
              best_move = i

  return best_move


### 1 if is_obstacle, 0 if not (if valid move)
def is_obstacle(snake_pos, snake_body):
  x, y = snake_pos

  # Check if the snake head is outside the grid boundaries (wall collision)
  if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
      return 1

  # Check if the snake head collides with its body
  for block in snake_body[1:]: # ignore head 
      if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
          return 1

  return 0







### State: snake position (x, y), food_position (x, y), obstacles (up, down, left, right)
def get_state(snake_pos, snake_body, direction, food_pos):
  """
  actions = ["UP", "DOWN", "LEFT", "RIGHT"]
  snake_x = snake_pos[0]
  snake_y = snake_pos[1]
  
  food_x = food_pos[0]
  food_y = food_pos[1]
  
  obstacle_up = is_obstacle([snake_x, snake_y - BLOCK_SIZE], snake_body)
  obstacle_down = is_obstacle([snake_x, snake_y + BLOCK_SIZE], snake_body)
  obstacle_left = is_obstacle([snake_x - BLOCK_SIZE, snake_y], snake_body)
  obstacle_right = is_obstacle([snake_x + BLOCK_SIZE, snake_y], snake_body)
  direction_int = actions.index(direction)
  return [snake_x, snake_y, food_x, food_y, direction_int, obstacle_up, obstacle_down, obstacle_left, obstacle_right]
  """
  """
  actions = ["UP", "DOWN", "LEFT", "RIGHT"]
  snake_x = snake_pos[0]
  snake_y = snake_pos[1]
  
  food_x = food_pos[0]
  food_y = food_pos[1]

  block_up = [snake_x, snake_y - BLOCK_SIZE]
  block_down = [snake_x, snake_y + BLOCK_SIZE]
  block_left = [snake_x - BLOCK_SIZE, snake_y]
  block_right = [snake_x + BLOCK_SIZE, snake_y]

  dir_up = direction == "UP"
  dir_down = direction == "DOWN"
  dir_left = direction == "LEFT"
  dir_right = direction == "RIGHT"
  
  # obstacle relative to snake move direction
  obstacle_straight = (dir_up and is_obstacle(block_up, snake_body)) or (dir_left and is_obstacle(block_left, snake_body)) or (dir_right and is_obstacle(block_right, snake_body)) or (dir_down and is_obstacle(block_down, snake_body))
  obstacle_left = (dir_up and is_obstacle(block_left, snake_body)) or (dir_left and is_obstacle(block_down, snake_body)) or (dir_right and is_obstacle(block_up, snake_body)) or (dir_down and is_obstacle(block_right, snake_body))
  obstacle_right = (dir_up and is_obstacle(block_right, snake_body)) or (dir_left and is_obstacle(block_up, snake_body)) or (dir_right and is_obstacle(block_down, snake_body)) or (dir_down and is_obstacle(block_left, snake_body))
  
  obstacle_up = is_obstacle(block_up, snake_body)
  obstacle_down = is_obstacle(block_down, snake_body)
  obstacle_left = is_obstacle(block_left, snake_body)
  obstacle_right = is_obstacle(block_right, snake_body)


  food_up = food_y < snake_y
  food_down = food_y > snake_y
  food_left = food_x < snake_x
  food_right = food_x > snake_x
  state = [obstacle_up, obstacle_down, obstacle_left, obstacle_right, dir_up, dir_down, dir_left, dir_right, food_up, food_down, food_left, food_right]
  #print([obstacle_straight, obstacle_left, obstacle_right, dir_up, dir_down, dir_left, dir_right, food_up, food_down, food_left, food_right])
  #print(np.array(state, dtype=int))
  return np.array(state, dtype=int)
"""
if __name__ == "__main__":
    game_loop()  # Restart the game when the loop ends

