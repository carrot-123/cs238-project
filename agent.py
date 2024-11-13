from collections import defaultdict
import pyautogui
import pyperclip

def parseInventory(inventory):
  # read inventories into a dictionary
  inventory_dict = defaultdict(int)
  print(inventory)
  # symbols
  # read from first semicolon to "Items"
  symbols_index = inventory.index("Symbols")
  inventory = inventory[symbols_index:]
  colon_index = inventory.index(":")
  # CHECK!!! items not exist yet
  if "Items" in inventory:
    end_index = inventory.index("Items")
  elif "Destroyed" in inventory:
    end_index = inventory.index("Destroyed")
  # in case there are destroyed items
  else:
    end_index = len(inventory)
  
  symbols = inventory[colon_index + 1: end_index]
  symbols = symbols.split()
  print(symbols)
  # first numeric characters are the number
  for symbol in symbols:
    index = 0
    count = 0
    while symbol[index].isdigit():
      count += int(symbol[index])
      index += 1
    print(count)
    name = symbol[index:]
    inventory_dict[name] += count

  # items --> treat each word as an item
  # read from second semicolon to "Items"
  if "Items" in inventory:
    inventory = inventory[end_index + 1:]
    colon_index = inventory.index(":")
    if "Destroyed" in inventory:
      end_index = inventory.index("Destroyed")
    # in case there are destroyed items
    else:
      end_index = len(inventory)

    
    items = inventory[colon_index + 1:end_index]
    items = items.split()
    for item in items:
      inventory_dict[item] += 1
  print(inventory_dict)
  # first split str into a list from Symbols to Items
  # maybe I will just treat each word as an item


def playGame():
  pyautogui.PAUSE = 0.5
  #pyautogui.press('enter')
  #pyautogui.press('enter')
  # click enter
  # click enter(confirm)
  # start game

  # start game
  status = ""
  """
  pyautogui.press('enter')
  pyautogui.press("space") # spin
  
  coins = pyperclip.paste()
  print(coins)
  """
  
  # hardcode --> you can only have 10 items --> won't need to worry about a second line
  while status != "Retry":
    # just keep choosing items until you can spin
    pyautogui.press("i") # inventory
    inventory = pyperclip.paste()
    parseInventory(inventory)
    print("here")
    pyautogui.press("i") 
    

    pyautogui.press("space") # spin
    coins = pyperclip.paste()
    print(coins)
    

    
    pyautogui.press("down")
    pyautogui.press("left")
    pyautogui.press("1")
    added_item = pyperclip.paste()
    print(added_item)

    pyautogui.press("down")
    status = pyperclip.paste()
    while "Pay" in status or status == "Confirm" or status == "Retry":
      print(status)
      pyautogui.press("enter") # pay money (or retry)
      pyautogui.press("enter") # click confirm
      pyautogui.press("down")
      status = pyperclip.paste()

    while status != "SPIN":

      # same inventory, pick an item
      pyautogui.press("down")
      pyautogui.press("left")
      pyautogui.press("1")
      added_item = pyperclip.paste()
      # print(added_item)

      # check inventory
      pyautogui.press("i") # inventory
      inventory = pyperclip.paste()
      parseInventory(inventory)
      print("here")
      pyautogui.press("i") # inventory
      pyautogui.press("down")
      status = pyperclip.paste()
    
      # after paying rent, choose another action always
      # copy paste code from above

  # after first payment --> choose 2 items

  # click down
  # while item is not Deny
  # read item
  # update curr state space
  # move right
  # if item is Deny, confirm


  # click enter (spin)
  # screen read (Coin22) --> probably need to take into consideration how many spins are left?
  # look through 3 times (any key, left, right right)
  # pick random action (1, 2, 3, s)
  # read inventory
  # spin

def main():
  # text = pyperclip.paste()
  # print(text) 

  playGame()
  inventory = """New Email
    Payments: 0/12 
    Symbols ( 20 ):
    14Empty 1Coin 1Ore 1Flower 1Cherry 1Pearl 1Cat 

    Destroyed Symbols ( 1 ):
    1Milk """
  #parseInventory(inventory)

if __name__ == '__main__':
    main()

# run this 10 times, a run ends when you return to menu,
# maybe if you don't make it through all the way, you penalize the reward??
# read to a csv file
# diff state spaces will have different rewards...

# try to get through one game first

# click enter
# click enter(confirm)


# click i (inventory)
# click down
# while item is not Deny
# read item
# update curr state space
# move right
# if item is Deny, confirm


# click enter (spin)
# screen read (Coin22) --> probably need to take into consideration how many spins are left?
# look through 3 times (any key, left, right right)
# pick random action (1, 2, 3, s)
# read inventory
# spin
# get reward


# after paying rent --> you choose an item first

