from __future__ import annotations

import csv
import os
import random
import time
import requests
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .adaptor import ArcEngineAdaptor

# Helper to convert grid to 16‑char hex string (no spaces)

def grid_to_hex(grid: np.ndarray) -> str:
    # Flatten row‑major and map each value to a hex digit
    flat = grid.flatten()
    if flat.size > 16:
        # keep first 16 values for representation; real board may be larger but our prompt expects 16
        flat = flat[:16]
    return ''.join(f'{int(v):X}' for v in flat)

class OSSAgent:
    """Simplified agent that uses an :class:`ArcEngineAdaptor` and a local GPT‑OSS via Ollama.

    Parameters
    ----------
    game_id: str
        The ARC‑interactive game id.
    seed: int, default 0
    max_steps: int, default 100
    log_dir: str, default 'runs/oss_v1'
    """

    def __init__(self, game_id: str, seed: int = 0, max_steps: int = 100,
                 log_dir: str = "runs/oss_v1", ollama_url: str = "http://localhost:11434",
                 model: str = "gptoss"):
        self.game_id = game_id
        self.adaptor = ArcEngineAdaptor(game_id, seed)
        self.max_steps = max_steps
        self.log_dir = Path(log_dir)
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._ensure_log()

    def _ensure_log(self) -> None:
        self.log_path = self.log_dir / f"{self.game_id}_log.csv"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        # Write header if file does not exist or empty
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "step",
                    "hex_grid",
                    "available_actions",
                    "state",
                    "llm_prompt",
                    "llm_output",
                    "action_taken"
                ])
                writer.writeheader()

    def _build_prompt(self, hex_grid: str, available_actions: List[int], state_name: str | None) -> str:
        return f"""You are an ARC game solver.

Board: {hex_grid}
Available actions: {available_actions}
State: {state_name or 'UNKNOWN'}

Choose the next action.  Respond only with one line:
- A single integer ID (e.g., "1").
- For a click, `CLICK x y` where x and y are coordinates.
"""

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an ARC game solver."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 64,
            "stream": False
        }
        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            # extract assistant content
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            elif "message" in data:
                return data["message"].get("content", "").strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def _parse_llm_output(self, text: str, available_actions: List[int]) -> Tuple[int, Optional[int], Optional[int]]:
        """
        Robustly parse LLM output into an action ID and optional (x, y) coordinates.
        Handles variations like 'CLICK 2 3', 'CLICK(2,3)', or just '1'.
        """
        raw_txt = text.strip()
        upper_txt = raw_txt.upper()

        # Case 1: Check for CLICK action first (prefixed with CLICK)
        if "CLICK" in upper_txt:
            # Match "CLICK" followed by digits, possibly separated by spaces or punctuation
            match = re.search(r"CLICK\s*(\d+)\s*[,\s]*(\d+)", raw_txt, re.IGNORECASE)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                return 6, x, y
            
            # Fallback for cases where it's just "CLICK" followed by digits but without space/comma 
            # (though regex above usually catches this if spacing is consistent)
            raise ValueError(f"Detected CLICK action but could not parse coordinates from: '{text}'")

        # Case 2: Standard integer action selection
        # Find the first occurrence of a digit to handle "Action: 1", "The result is 4.", etc.
        match = re.search(r"(\d+)", raw_txt)
        if not match:
            raise ValueError(f"No numeric value found in LLM output: '{text}'")
        
        action_id = int(match.group(1))
        if action_id not in available_actions:
            raise ValueError(f"Action ID {action_id} is not among the available actions: {available_actions}")
        
        return action_id, None, None

    def run_episode(self) -> List[Tuple[int | str, int | None, int | None]]:
        actions: List[Tuple[int | str, int | None, int | None]] = []
        step_num = 0
        while step_num < self.max_steps:
            grid = self.adaptor.observe()
            hex_grid = grid_to_hex(grid)
            available_actions = self.adaptor.available_actions()
            state_name = self.adaptor.state or "UNKNOWN"
            prompt = self._build_prompt(hex_grid, available_actions, state_name)

            # Retry up to 2 times (total attempts = 3)
            attempt = 0
            success = False
            llm_output = None
            action_id = None; x=None; y=None
            while attempt < 3 and not success:
                try:
                    llm_output = self._call_llm(prompt)
                    action_id, x, y = self._parse_llm_output(llm_output, available_actions)
                    success = True
                except Exception:
                    attempt += 1
                    time.sleep(0.5)  # small backoff
            if not success:
                # log failure and fallback to random action
                action_id = random.choice(available_actions)
                llm_output = "FAILED"
            # Execute the step
            try:
                new_grid = self.adaptor.step(action_id, x, y)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to execute action {action_id}: {exc}") from exc

            # Log the step
            row = {
                "step": step_num,
                "hex_grid": hex_grid,
                "available_actions": str(available_actions),
                "state": state_name,
                "llm_prompt": prompt.replace("\n", " ")[:200],  # truncate for file
                "llm_output": llm_output,
                "action_taken": f"{action_id}" if action_id != 6 else f"CLICK {x} {y}"
            }
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

            actions.append((action_id, x, y))
            step_num += 1
            # Break if terminal state reached
            if self.adaptor.state in {"WIN", "GAME_OVER"}:
                break
        return actions

# End of module
