ARC-AGI-3 Quickstart

ARC-AGI-3 is an Interactive Reasoning Benchmark designed to measure an AI Agent’s ability to generalize in novel, unseen environments.

Traditionally, to measure AI, static benchmarks have been the yardstick. These work well for evaluating LLMs and AI reasoning systems. However, to evaluate frontier AI agent systems, we need new tools that measure:

    Exploration
    Percept → Plan → Action
    Memory
    Goal Acquisition
    Alignment

By building agents that can play ARC-AGI-3, you’re directly contributing to the frontier of AI research.
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
Each game provides an explicit set of available actions. The actions available vary per game and are stated explicitly so your agent knows what it can do. To learn about the standardized action interface, see the Actions page. To see how to retrieve a game’s available actions programmatically, see List Available Actions.
Available Games

List of ARC-AGI-3 games and how to discover them
ARC-AGI-3 consists of a series of public games that are playable by both humans and AI agents.
​
Discovering Games
To see a list of available games:

    Browse games at three.arcprize.org
    Use the ARC-AGI Toolkit to list games programmatically

By default, three games are available to anonymous users after launch. An API key is required to access the remaining public games. Get a free API key to unlock them.
​
Example Games

    ls20 - Agent reasoning
    ft09 - Elementary Logic
    vc33 - Orchestration

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

Agents
LLM Agents

Language model agents for ARC-AGI-3.
​
LLM Agent

    Standard OpenAI API agent that observes the game state and chooses actions using function calling, maintains conversation history with 10-message limit.
    Default Model: gpt-4o-mini
    Usage: --agent=llm

​
Fast LLM Agent

    Skips the observation step entirely (DO_OBSERVATION=False), making decisions faster but potentially less informed - trades accuracy for speed.
    Default Model: gpt-4o-mini
    Usage: --agent=fastllm

​
ReasoningLLM

    Uses OpenAI’s o4-mini model and captures detailed reasoning metadata including reasoning tokens and thought process in the action.reasoning field.
    Default Model: o4-mini
    Usage: --agent=reasoningllm

​
GuidedLLM

    Uses the most advanced o3 model with high reasoning effort and includes explicit game-specific rules/strategy in the prompt. This template is for education purposes only, it won’t generalize to other games.
    Default Model: o3
    Usage: --agent=guidedllm

​
Example Usage

# Run LLM agent on a specific game
uv run main.py --agent=llm --game=ls20

# Run fast LLM agent on all games
uv run main.py --agent=fastllm

​
Benchmarking your LLM agent
If you are comparing prompts, model versions, or agent architectures, use the benchmarking tooling to produce repeatable scorecards and replays. It is designed to get you from zero to benchmarking quickly and works well alongside the LLM agent templates.

    Benchmarking Tooling (BETA)

​
Handling Malformed Outputs
LLM agents are expected to return exactly one of the valid action names (RESET, ACTION1 – ACTION6). In the reference implementation we simply call .strip() on the model response and forward the resulting string. In practice a model might return an empty string, additional commentary, or a token that is not a valid action. When that happens the agent will raise a ValueError and the current game will terminate. To make your agent more robust you can:

    Post-process the model output – e.g. extract the first word that looks like an action using a regular expression.
    Fallback to a safe action – if parsing fails, choose a random valid action or repeat the previous one.
    Log the bad response in the reasoning field – this makes debugging much easier when you review the replay in the UI.

AgentOps Template

AgentOps Template
AgentOps provides observability and debugging tools for AI agents. These templates are designed to help you get the most out of their platform.
​
ReasoningAgent
This template provides a reasoning agent that maintains a history of its actions and observations to build a hypothesis about the game’s rules. It’s fully integrated with AgentOps for detailed tracing and debugging, giving you more information about your agent’s performance. To run this agent:

uv run main.py --agent=reasoningagent --game=ls20

Source File
​
AgentOps Tracing
When you install the agentops package and set your API key, all agents are automatically decorated with @trace_agent_session. This streams detailed traces of your agent’s execution to your AgentOps dashboard.

HuggingFace Template

HuggingFace Template
These templates provide a great starting point for building agents with the HuggingFace ecosystem. Check out the smolagents documentation to learn more.

    SmolCodingAgent: A text-based agent that uses a code-generating model to reason about the game and execute actions as Python code.
    SmolVisionAgent: A multimodal agent that processes game frames as images, allowing it to “see” the game state.


Create Your Own Agent

Build a custom agent for ARC-AGI-3 games
Create AI agents that can play ARC-AGI-3 games by implementing the required interface methods. The following is based off the ARC-AGI-3 Agents repo.
​
Step 1: Create Your Agent File
First, head over to the ARC-AGI-3-Agents repo and clone it

git clone https://github.com/arcprize/ARC-AGI-3-Agents.git

Make sure you have your ARC_API_KEY populated in your environment variables. You can obtain this key by signing up for an account on the ARC-AGI-3 website. Next, create a new Python file for your agent inside the agents/ directory. For this example, let’s copy the random_agent.py template.

cp agents/templates/random_agent.py agents/my_awesome_agent.py

Now, modify agents/my_awesome_agent.py and rename the class to MyAwesomeAgent.

# agents/my_awesome_agent.py

from .agent import Agent # Make sure to change from `..` imports
from .structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
import random

# Rename the class
class MyAwesomeAgent(Agent):
    """A simple agent that chooses random actions."""

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Your logic to determine if the game is finished
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Your custom decision-making logic goes here
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Start or restart the game
            action = GameAction.RESET
        else:
            # Choose a random action (except RESET)
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        # Add reasoning for simple actions
        if action.is_simple():
            action.reasoning = f"Chose {action.value} randomly"
        # For complex actions, set coordinates
        elif action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })
            action.reasoning = {"action": action.value, "reason": "Random choice"}

        return action

​
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

​
Agent Not Found Errors
If you see ValueError: Agent '<your-agent>' not found, double-check the following:

    Your agent class is correctly located in the agents directory (or a subdirectory).
    The class name is correctly spelled and matches the name you provided to the --agent flag (in lower case).
    You have saved your changes to your agent file.


Local vs Online

Playing games locally with the engine vs online via the API
Choose how you want to run ARC-AGI-3 games.
Local (Recommended)
Fast, no rate limits, run many instances
Online
Scorecards, replays, requires API key
​
Local
Run games locally using the ARC-AGI engine. This is the recommended approach for development and testing.

from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode="terminal")

Advantages	Limitations
~2,000 FPS (120,000 frames per minute)	No online scorecards
No rate limits	No shareable replays
Run as many instances as you want	
No API key required	
​
Online
Run games via the API to get scorecards and replays.

from arc_agi import Arcade, OperationMode

arc = Arcade(operation_mode=OperationMode.ONLINE)
env = arc.make("ls20", render_mode="terminal")

Advantages	Limitations
View scorecards online	Requires API key
Shareable replays	Capped at 600 requests per minute
Results appear on leaderboard	

warms

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

