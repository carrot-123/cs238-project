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

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
     
  
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
        reward = -10
        done = True
        # start over
  for block in snake_body[1:]:
      if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
        reward = -10
        done = True

  next_state = get_state(snake_pos, snake_body, direction, food_pos)
  
  return next_state, reward, done

def play():
  global snake_pos
  global snake_body
  global direction
  global food_pos
  global change_to
  global food_spawn

  model = DQN(12, 4)
  model.load_state_dict(torch.load("snake.pth", weights_only=True))
  model.eval()
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
    state = get_state(snake_pos, snake_body, direction, food_pos)
    done = False
    while not done:
      count += 1
      if (count > 1000): # prevent infinite loop
        done = True
        break
        
      action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()  # Choose action based on Q-values
      state, reward, done = step(action, done)  # Take the action and get the new state
      if reward == 10:
        count = 0
      score += reward
      
        
    scores.append(score)
  np.savetxt("scores.txt", np.array(scores, dtype=int))
  print("mean score: " + str(np.mean(scores)))
  print("median score: " + str(np.median(scores)))
  print("min score: " + str(np.min(scores)))
  print("max score: " + str(np.max(scores)))

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

### State: snake position (x, y), food_position (x, y), obstacles (up, down, left, right)
def get_state(snake_pos, snake_body, direction, food_pos):
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
  
  obstacle_up = is_obstacle(block_up, snake_body)
  obstacle_down = is_obstacle(block_down, snake_body)
  obstacle_left = is_obstacle(block_left, snake_body)
  obstacle_right = is_obstacle(block_right, snake_body)

  food_up = food_y < snake_y
  food_down = food_y > snake_y
  food_left = food_x < snake_x
  food_right = food_x > snake_x
 
  state = [obstacle_up, obstacle_down, obstacle_left, obstacle_right, dir_up, dir_down, dir_left, dir_right, food_up, food_down, food_left, food_right]
  return np.array(state, dtype=int)

if __name__ == "__main__":
    play()


