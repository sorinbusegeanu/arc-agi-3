Game Schema

Structure and format of ARC-AGI-3 game environments
ARC-AGI-3 games are turn-based environments where agents interact with 2D grids through a standardized action interface. Each game maintains state through discrete action-response cycles.

    Agents receive 1-N frames of JSON objects with the game state and metadata.
    Agents respond with an action to interact with the game.

​
Grid Structure

    Dimensions: Maximum 64x64 grid size
    Cell Values: Integer values 0-15 representing different states/colors
    Coordinate System: (0,0) at top-left, (x,y) format

​
Game ID Format
Game IDs are formatted as <game_name>-<version>. game_names are stable, but version may change as games update.
​
Game Available Actions
Each game provides an explicit set of available actions. The actions available vary per game and are stated explicitly so your agent knows what it can do. 

Actions

Input interface for ARC-AGI-3 games
All games implement a standardized action interface with seven core actions:
Action	Description
RESET	Initialize or restarts the game/level state
ACTION1	Simple action - varies by game (semantically mapped to up)
ACTION2	Simple action - varies by game (semantically mapped to down)
ACTION3	Simple action - varies by game (semantically mapped to left)
ACTION4	Simple action - varies by game (semantically mapped to right)
ACTION5	Simple action - varies by game (e.g., interact, select, rotate, attach/detach, execute, etc.)
ACTION6	Complex action requiring x,y coordinates (0-63 range)
ACTION7	Simple action - Undo (e.g., interact, select)
​
Human Player Keybindings
When playing games manually in the ARC-AGI-3 UI, you can use these keyboard shortcuts instead of clicking action buttons:
Control Scheme	ACTION1	ACTION2	ACTION3	ACTION4	ACTION5	ACTION6	ACTION7
WASD + Space	W	S	A	D	Space	Mouse Click	CTRL/CMD+Z
Arrow Keys + F	↑	↓	←	→	F	Mouse Click	CTRL/CMD+Z
All control schemes support mouse clicking for ACTION6 (coordinate-based actions). Choose whichever scheme feels most comfortable for your playstyle.
​
Available Actions
Each game explicitly defines the set of available actions that can be used within that game. This approach ensures clarity for both human and AI participants by making it clear which actions are permitted, thereby reducing confusion. In the human-facing UI, available actions are visually highlighted or dismissed to provide the same affordance. For each action taken, the metadata of the returned frame will indicate which actions are available. Agents may use this information to narrow the action space and develop effective strategies for completing the game. Note: Action 6 does not provide explicit X/Y coordinates for active areas. If Action 6 is available, only its availability will be indicated, without specifying which coordinates are active.


List Available Actions

How to retrieve available actions for a game using the ARC-AGI Toolkit
​
Overview
Each ARC-AGI game has its own set of available actions. Before interacting with a game, you can check which actions are supported using the action_space property.
​
Listing Actions for a Game
Once you’ve created an environment, use action_space to see what actions are available:

import arc_agi

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

# Get available actions
actions = env.action_space
for action in actions:
    print(action.name)

​
Understanding Action Information
Each action in the list is a GameAction object. You can inspect its properties:

actions = env.action_space

for action in actions:
    print(f"Name: {action.name}")
    print(f"Is complex: {action.is_complex()}")
    print("---")

Complex actions (like ACTION6) require additional data such as x,y coordinates when called.
​
Actions Update Each Step
The available actions can change as you play. After each step, check action_space for the current set of valid actions:

from arcengine import GameAction

# Initial actions
print(f"Initial actions: {[a.name for a in env.action_space]}")

# Take an action
obs = env.step(GameAction.ACTION1)

# Actions may have changed
print(f"Current actions: {[a.name for a in env.action_space]}")

​
Using Actions in Your Agent
Below is a an example of picking a random action from those available:

import random
from arcengine import GameAction

actions = env.action_space

# Choose a random available action
action = random.choice(actions)

# Handle complex actions that need coordinates
action_data = {}
if action.is_complex():
    action_data = {"x": 32, "y": 32}

obs = env.step(action, data=action_data)

