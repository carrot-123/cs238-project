
import time
import random

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
    # Update direction
  
  #print(str_action)
  
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
      print("here")
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



def train():
  global snake_pos
  global snake_body
  global direction
  global food_pos
  global change_to
  global food_spawn

  experience_replay = []
  state_dim = 12
  action_dim = 4
  lr = 0.0001
  discount = 0.99
  model = DQN(state_dim, action_dim)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  load = True
  num_episodes = 2000
  batch_size = 32
  epsilon = 1
  
  ### load model? --> check if this works
  if (load):
    checkpoint = torch.load('snake.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()

  scores = []
  for episode in range(num_episodes):
   
    snake_pos = [200, 200]
    snake_body = [[200, 200]]#[[80, 20], [60, 20], [40, 20]]

    # Direction variables
    direction = 'RIGHT'
    change_to = direction

    # Initial score
    score = 0

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
        # take random action with 0.1 chance
        if np.random.rand() <= epsilon:
          action = random.randint(0, 3)
        else:
          action = torch.argmax(model(torch.tensor(state, dtype=torch.float32))).item()  # Choose action based on Q-values
        
        # nothing looks changed?
        next_state, reward, done = step(action, done)  # Take the action and get the new state
        experience_replay.append((state, action, reward, next_state, done))  # Store in memory
        score += reward
        # Train the model using experience replay
        #print(batch_size)
        if len(experience_replay) > batch_size:
            
            minibatch = random.sample(experience_replay, batch_size)
            check = 0
            for s, a, r, next_s, d in minibatch:
              target = r
              check += 1
              if not d:
                target = r + discount * torch.max(model(torch.tensor(next_s, dtype=torch.float32))).item()

             
              
              target_f = model(torch.tensor(s, dtype=torch.float32)).detach().numpy()
              target_f[a] = target
              
              # model.fit(s, target)
              optimizer.zero_grad()
              loss = nn.MSELoss()(torch.tensor(target_f), model(torch.tensor(s, dtype=torch.float32)))

              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
              optimizer.step()
              
          
                
                
      
              state = next_state  # Move to the new state
              #if check >= 10 and loss > prev_loss:
              #  break
              #prev_loss = loss
              #checkpoint = {
              #'model_state_dict': model.state_dict(),
              #'optimizer_state_dict': optimizer.state_dict(),
              #'loss': loss,
              #}
              #torch.save(checkpoint, 'snake.pth')
              #print(f"Episode: {episode}, Loss: {loss.item()}")
        if epsilon > 0.01:
          epsilon *= 0.999
        #lr /= 2
    scores.append(score)
    print(f"SCORE: {np.mean(scores)}")
  torch.save(model.state_dict(), "snake.pth")
    
      

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
  #return [snake_x, snake_y, food_x, food_y, direction_int, obstacle_up, obstacle_down, obstacle_left, obstacle_right]

if __name__ == "__main__":
    train()


