import time
import random
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Screen dimensions
WIDTH, HEIGHT = 400, 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Block size for the snake and food
BLOCK_SIZE = 20
     
  
global snake_pos
global snake_body
global direction
global food_pos
global change_to
global food_spawn

def step(action, done):
  global snake_pos
  global snake_body
  global direction
  global food_pos
  global change_to
  global food_spawn

  actions = ["UP", "DOWN", "LEFT", "RIGHT"]
  str_action = actions[action]

  # Update direction
  if str_action == "UP" and direction != 'DOWN':
      change_to = 'UP'
  elif str_action == "DOWN" and direction != 'UP':
      change_to = 'DOWN'
  elif str_action == "LEFT" and direction != 'RIGHT':
      change_to = 'LEFT'
  elif str_action == "RIGHT" and direction != 'LEFT':
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
      reward = 10
      food_spawn = False
     
  else:
      # Remove the last segment if no food eaten
      snake_body.pop()
      reward = 0

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

  if snake_pos[0] < 0 or snake_pos[0] >= WIDTH or snake_pos[1] < 0 or snake_pos[1] >= HEIGHT:
        reward = 0
        done = True
        # start over
  for block in snake_body[1:]:
      if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
        reward = 0
        done = True
  
  return [], reward, done

def play():
  global snake_pos
  global snake_body
  global direction
  global food_pos
  global change_to
  global food_spawn

  num_episodes = 100

  scores = []
  for episode in tqdm.tqdm(range(num_episodes)):
   
    snake_pos = [220, 220]
    snake_body = [[220, 220]]

    # Direction variables
    direction = 'RIGHT'
    change_to = direction

    # Initial score
    score = 0
    count = 0

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
    while not done:
      action = greedy_move(snake_pos, snake_body, food_pos)
      state, reward, done = step(action, done)  # Take the action and get the new state
      score += reward
    scores.append(score)
  np.savetxt("scores.txt", np.array(scores, dtype=int))
  print("mean score: " + str(np.mean(scores)))
  print("median score: " + str(np.median(scores)))
  print("min score: " + str(np.min(scores)))
  print("max score: " + str(np.max(scores)))
  print("----------")
  
### State, Action, and Reward ###
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

if __name__ == "__main__":
    play()


