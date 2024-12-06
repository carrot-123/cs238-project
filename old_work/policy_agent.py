### rename to agent explore or agent train
from collections import defaultdict
import pyautogui
import pyperclip
import csv
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

item_dict = {
    "Amethyst": 0,
    "Anchor": 1,
    "Apple": 2,
    "Banana": 3,
    "Banana Peel": 4,
    "Bar of Soap": 5,
    "Bartender": 6,
    "Bear": 7,
    "Beastmaster": 8,
    "Bee": 9,
    "Beehive": 10,
    "Beer": 11,
    "Big Ore": 12,
    "Big Urn": 13,
    "Billionaire": 14,
    "Bounty Hunter": 15,
    "Bronze Arrow": 16,
    "Bubble": 17,
    "Buffing Capsule": 18,
    "Candy": 19,
    "Card Shark": 20,
    "Cat": 21,
    "Cheese": 22,
    "Chef": 23,
    "Chemical Seven": 24,
    "Cherry": 25,
    "Chick": 26,
    "Chicken": 27,
    "Clubs": 28,
    "Coal": 29,
    "Coconut": 30,
    "Coconut Half": 31,
    "Coin": 32,
    "Comedian": 33,
    "Cow": 34,
    "Crab": 35,
    "Crow": 36,
    "Cultist": 37,
    "Dame": 38,
    "Diamond": 39,
    "Diamonds": 40,
    "Diver": 41,
    "Dog": 42,
    "Dove": 43,
    "Dud": 44,
    "Dwarf": 45,
    "Egg": 46,
    "Eldritch Creature": 47,
    "Emerald": 48,
    "Empty": 49,
    "Essence Capsule": 50,
    "Farmer": 51,
    "Five Sided Die": 52,
    "Flower": 53,
    "Frozen Fossil": 54,
    "Gambler": 55,
    "General Zaroff": 56,
    "Geologist": 57,
    "Golden Arrow": 58,
    "Golden Egg": 59,
    "Goldfish": 60,
    "Golem": 61,
    "Goose": 62,
    "Hearts": 63,
    "Hex of Destruction": 64,
    "Hex of Draining": 65,
    "Hex of Emptiness": 66,
    "Hex of Hoarding": 67,
    "Hex of Midas": 68,
    "Hex of Tedium": 69,
    "Hex of Thievery": 70,
    "Highlander": 71,
    "Honey": 72,
    "Hooligan": 73,
    "Hustling Capsule": 74,
    "Item Capsule": 75,
    "Jellyfish": 76,
    "Joker": 77,
    "Key": 78,
    "King Midas": 79,
    "Light Bulb": 80,
    "Lockbox": 81,
    "Lucky Capsule": 82,
    "Magic Key": 83,
    "Magpie": 84,
    "Martini": 85,
    "Matryoshka Doll": 86,
    "Matryoshka Doll 2": 87,
    "Matryoshka Doll 3": 88,
    "Matryoshka Doll 4": 89,
    "Matryoshka Doll 5": 90,
    "Mega Chest": 91,
    "Midas Bomb": 92,
    "Milk": 93,
    "Mine": 94,
    "Miner": 95,
    "Missing": 96,
    "Monkey": 97,
    "Moon": 98,
    "Mouse": 99,
    "Mrs Fruit": 100,
    "Ninja": 101,
    "Omelette": 102,
    "Orange": 103,
    "Ore": 104,
    "Owl": 105,
    "Oyster": 106,
    "Peach": 107,
    "Pear": 108,
    "Pearl": 109,
    "Pirate": 110,
    "Pinata": 111,
    "Present": 112,
    "Pufferfish": 113,
    "Rabbit": 114,
    "Rabbit Fluff": 115,
    "Rain": 116,
    "Removal Capsule": 117,
    "Reroll Capsule": 118,
    "Robin Hood": 119,
    "Ruby": 120,
    "Safe": 121,
    "Sand Dollar": 122,
    "Sapphire": 123,
    "Seed": 124,
    "Shiny Pebble": 125,
    "Silver Arrow": 126,
    "Sloth": 127,
    "Snail": 128,
    "Spades": 129,
    "Spirit": 130,
    "Strawberry": 131,
    "Sun": 132,
    "Target": 133,
    "Tedium Capsule": 134,
    "Thief": 135,
    "Three Sided Die": 136,
    "Time Capsule": 137,
    "Toddler": 138,
    "Tomb": 139,
    "Treasure Chest": 140,
    "Turtle": 141,
    "Urn": 142,
    "Void Creature": 143,
    "Void Fruit": 144,
    "Void Stone": 145,
    "Watermelon": 146,
    "Wealthy Capsule": 147,
    "Wildcard": 148,
    "Wine": 149,
    "Witch": 150,
    "Wolf": 151
}

# load model
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class TrainAgent:
  def __init__(self, state_dim, action_dim, lr, discount):
   

    self.state_dim = state_dim
    self.action_dim = action_dim
    self.lr = lr
    self.discount = discount
    self.model = DQN(state_dim, action_dim)
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.load = true

  
    ### load model? --> check if this works
    if (self.load):
      checkpoint = torch.load('checkpoint.pth', weights_only=True)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      model.eval()
    ### 

  # need to check if this is correct!
  """
  def qLearning(self, s, a, r, s_prime): # write to a csv file which will be used by another agent to properly play the game
    u = torch.max(self.model(torch.tensor(s_prime, dtype=torch.float32))).item()
    target = r + self.discount * u
    target_f = self.model(torch.tensor(s, dtype=torch.float32)).detach().numpy()
    target_f[a] = target
    self.optimizer.zero_grad()
    loss = nn.MSELoss()(torch.tensor(target_f, dtype=torch.float32), self.model(torch.tensor(s, dtype=torch.float32)))
    
    loss.backward()
    print("loss: " + str(loss)) # graph loss
    self.optimizer.step()
    # save model here?

    checkpoint = {
    'model_state_dict': this.model.state_dict(),
    'optimizer_state_dict': this.optimizer.state_dict(),
    'loss': loss,
    # you may add other information to add 
    }
    torch.save(checkpoint, 'checkpoint.pth')
    """

state_dim = 152
action_dim = 153 # all symbols plus "skip"
agent = TrainAgent(state_dim, action_dim, lr=0.1, discount=0.9)

def parseInventory(inventory):
  # read inventories into a dictionary
  
  inventory_dict = defaultdict(int)

  # symbols
  # read from first semicolon to "Items"
  symbols_index = inventory.index("Symbols")
  inventory = inventory[symbols_index:]
  colon_index = inventory.index(":")
  
  if "Items" in inventory:
    end_index = inventory.index("Items")
  elif "Destroyed" in inventory:
    end_index = inventory.index("Destroyed")
  else:
    end_index = len(inventory)
  
  symbols = inventory[colon_index + 1: end_index]
  #symbols = symbols.split()
  
  # first numeric characters are the number
  pattern = r"(\d+)([a-zA-Z\s]+)"
  symbols = symbols.replace(".", "")
  symbols = symbols.replace("-", " ")
  symbols = symbols.replace("ñ", "n")
    # Find all matches for the pattern in the input string
  matches = re.findall(pattern, symbols)
  
  for number_str, item in matches:
    item = item.strip()
    # Convert the number to an integer
    number = int(number_str)
    # Add the number to the corresponding item in the defaultdict
    inventory_dict[item] += number
  print("dict: " + str(inventory_dict))
  return inventory_dict

def parseItem(added_item): #implement to clean up code more
  if "Common" in added_item:
    added_item = added_item[:added_item.index("Common")]
  elif "Uncommon" in added_item:
    added_item = added_item[:added_item.index("Uncommon")]
  elif "Rare" in added_item:
    added_item = added_item[:added_item.index("Rare")]
  elif "Super Rare" in added_item:
    added_item = added_item[:added_item.index("Super Rare")]
  
  added_item = added_item.strip()
  added_item = added_item.replace("-", " ")
  added_item = added_item.replace(".", "")
  added_item = added_item.replace("\n", " ")
  added_item = added_item.replace("ñ", "n")
  return added_item

def playGame():
  global item_dict
  pyautogui.PAUSE = 0.5

  # start game
  status = ""

  i = 0
  print("hello")
  added_item = ""
  while status != "Retry": # change this to main menu????
    added_item = ""
    pyautogui.press("i") # inventory
    inventory = pyperclip.paste()
    temp = parseInventory(inventory) # current state, and next state
    s_prime = [0 for _ in range(state_dim)]
    for item, count in temp.items():
      item = item.replace("\n", " ")
      item_index = item_dict[item]
      s_prime[item_index] = count
    
    if i != 0: # skip first spin's reward, need to check if this logic is correct
      # update neural network here
      # call update function
      
      print(s_prime)
      # convert 
      
      print("s: " + str(s))
      print("a: " + str(a))
      print("r: " + str(r))
      print("s_prime: " + str(s_prime))
      a = a.replace("\n", " ")
      agent.qLearning(s, item_dict[a], r, s_prime)
        # pass to network haha
    s = s_prime

    i += 1
    
    pyautogui.press("i") 
    # add a limit so that you are always carrying 21 items?
    pyautogui.press("space") # spin
    coins = pyperclip.paste()
    if i != 0: # skip first spin's reward
      r = int(coins[len("Coin"):coins.index("\n")]) # reward for previous action
      print(r)
      
      print("CHOOSING2")
      
      pyautogui.press("down")
      added_item = pyperclip.paste()
      print("added_item4: " + added_item)
      if "\n" in added_item:
        added_item = parseItem(added_item)
        if added_item in item_dict.keys():
          # change later to exploration strategy
          pyautogui.press("left")
          added_item = pyperclip.paste()
          added_item = parseItem(added_item)
          a = added_item
          pyautogui.press("1")

        else:
          print("SKIPPING!")
          pyautogui.press("s") #skip
      ###
  
       

    pyautogui.press("down")
    status = pyperclip.paste()
    print("STATUS: " + status)
    # need it to endthe game when it hits floor 10!!! replay the level!!!!
    # exit to main menu and play again?
    # unclear if this works yet
    if "Endless" in status:
      i = 0
      pyautogui.press("down") # navigate to main menu
      pyautogui.press("enter")
      # a pop up that you are in endless mode?
      pyautogui.press("enter") # not sure if this is the right key

      pyautogui.press("up") # navigate to level selection
      pyautogui.press('enter')

      pyautogui.press("down") # navigate to floor 1
      pyautogui.press("left")
      pyautogui.press("up")
      pyautogui.press("enter")
      games_won = 0
      
    while "Pay" in status or status == "Confirm" or status == "Main Menu" or status == "Skip":
      if status == "Main Menu":
        i = 0
        pyautogui.press("up")
        pyautogui.press("enter")
      else:
        pyautogui.press("enter") # pay money or retry
        pyautogui.press("enter") # click confirm
      pyautogui.press("down") # check current status
      status = pyperclip.paste()

   
    while status != "SPIN" and status != "Retry":
      added_item = ""
      print("CHOOSING")
      pyautogui.press("left")
      added_item = pyperclip.paste()
      print("aaaa: " + added_item)
      # if "\n" in added_item:
    
      added_item = added_item[:added_item.index("\n")]
      added_item = parseItem(added_item)

      if added_item in item_dict.keys():
        # change later to exploration strategy
        pyautogui.press("left")
        added_item = pyperclip.paste()
       
        added_item = parseItem(added_item)
        a = added_item
        pyautogui.press("1")
      else:
        print("SKIPPING!")
        pyautogui.press("s") #skip

      pyautogui.press("left") # check status????
      status = pyperclip.paste()
      print(status)

def main():
  playGame()
  
if __name__ == '__main__':
    main()