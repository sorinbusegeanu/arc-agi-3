ARC-AGI-3 is an Interactive Reasoning Benchmark designed to measure an AI Agent’s ability to generalize in novel, unseen environments.

Traditionally, to measure AI, static benchmarks have been the yardstick. These work well for evaluating LLMs and AI reasoning systems. However, to evaluate frontier AI agent systems, we need new tools that measure:

    Exploration
    Percept → Plan → Action
    Memory
    Goal Acquisition
    Alignment

.
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
three games are available to anonymous users after launch. An API key is required to access the remaining public games. Get a free API key to unlock them.
​
Example Games

    ls20 - Agent reasoning
    ft09 - Elementary Logic
    vc33 - Orchestration

Actions


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

​
Available Actions
Each game explicitly defines the set of available actions that can be used within that game. 
For each action taken, the metadata of the returned frame will indicate which actions are available. 
Agents may use this information to narrow the action space and develop effective strategies for completing the game. 
Note: Action 6 does not provide explicit X/Y coordinates for active areas. If Action 6 is available, only its availability will be indicated, without specifying which coordinates are active.

# Run fast LLM agent on all games
uv run main.py --agent=fastllm

​


 Benchmarking Tooling (BETA)

​
Handling Malformed Outputs
LLM agents are expected to return exactly one of the valid action names (RESET, ACTION1 – ACTION6). In the reference implementation we simply call .strip() on the model response and forward the resulting string. In practice a model might return an empty string, additional commentary, or a token that is not a valid action. When that happens the agent will raise a ValueError and the current game will terminate. To make your agent more robust you can:

    Post-process the model output – e.g. extract the first word that looks like an action using a regular expression.
    Fallback to a safe action – if parsing fails, choose a random valid action or repeat the previous one.
    Log the bad response in the reasoning field – this makes debugging much easier when you review the replay in the UI.

ReasoningAgent
This template provides a reasoning agent that maintains a history of its actions and observations to build a hypothesis about the game’s rules. It’s fully integrated with AgentOps for detailed tracing and debugging, giving you more information about your agent’s performance. To run this agent:

uv run main.py --agent=reasoningagent --game=ls20

Now, modify agents/my_awesome_agent.py and rename the class to MyAwesomeAgent.

# agents/my_awesome_agent.py

from .agent import Agent # Make sure to change from `..` imports
Step 2: Register Your Agent
To make your agent available to run, add an import statement to agents/__init__.py and add it to the AVAILABLE_AGENTS dictionary:

# agents/__init__.py
# ... existing imports ...
from .my_awesome_agent import MyAwesomeAgent

__all__ = [
    # ... existing agents ...
    "MyAwesomeAgent",
    "AVAILABLE_AGENTS",
]

​
Step 3: Run Your Agent
Your agent is now registered and ready to run. Use the class name in lower case as the value for the --agent argument.

# Run your custom agent on the 'ls20' game
uv run main.py --agent=myawesomeagent --game=ls20

You can also run it against all available games:

# Run your agent on all games
uv run main.py --agent=myawesomeagent

The replay of your agent is available at the end of the run. Make sure to watch your agent at play.
​
Troubleshooting
​
Relative Import Errors
If you move an agent file or create a new one outside the agents/ directory, you may encounter ImportError exceptions related to relative imports. Solution: Ensure your import statements use the correct relative pathing. The .. prefix goes up one directory level. For example, if your agent is in agents/my_agents/my_file.py, the imports should look like this:

# agents/my_agents/my_file.py

# Correct: Go up one level to the 'agents' package root
from ..agent import Agent
from ..structs import FrameData

# Incorrect: Assumes the file is in the 'agents' root
# from .agent import Agent

​Run games locally using the ARC-AGI engine. This is the recommended approach for development and testing.

from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode="terminal")


Swarms

Orchestrate agents across multiple games.
Swarms are used to orchestrate your agent across multiple games simultaneously.

git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

Each swarm:

    Creates one agent instance per game
    Runs all agents concurrently using threads
    Automatically manages scorecard opening and closing
    Handles cleanup when all agents complete
    Provides a link to view replay online

​
Running the Agent Swarm
The agent swarm is executed through main.py, which manages agent execution across multiple games with automatic scorecard tracking.
​
Swarm Command

uv run main.py --agent <agent_name> [--game <game_filter>] [--tags <tag_list>]

​
CLI Arguments
Argument	Short	Required	Description
--agent	-a	Yes	Choose which agent to run. Available agents can be found in the agents/ directory.
--game	-g	No	Filter games by ID prefix. Can be comma-separated for multiple filters (e.g., ls20,ft09). If not specified, the agent plays all available games.
--tags	-t	No	Comma-separated list of tags for the scorecard (e.g., experiment,v1.0). Tags help categorize and track different agent runs. Helpful when you want to compare different agents. Tags will be recorded on your scorecards.
​
Examples

# Run the random agent on all games
uv run main.py --agent=random

# Run an LLM agent on only the ls20 game
uv run main.py --agent=llm --game=ls20

# Run with custom tags for tracking
uv run main.py --agent=llm --tags="experiment,gpt-4,baseline"

# Run against an explicit list of games
uv run main.py --agent=random --game="ls20,ft09"

