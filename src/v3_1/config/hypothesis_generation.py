from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HypothesisGenerationSection:
    enable_deterministic: bool = True
    deterministic_rule_window_steps: int = 8
    deterministic_lag_min_steps: int = 0
    deterministic_lag_max_steps: int = 4
    deterministic_support_threshold: int = 1
    deterministic_contradiction_threshold: int = 2
    enable_llm: bool = False
    llm_provider: str = "openai_compat"
    llm_model_name: str = "Qwen/Qwen3.5-9B"
    llm_base_url: str = ""
    llm_api_key_env: str = ""
    llm_timeout_sec: float = 60.0
    llm_connect_timeout_sec: float = 5.0
    llm_read_timeout_sec: float = 60.0
    llm_retry_limit: int = 0
    llm_max_output_tokens: int = 1200
    llm_temperature: float = 0.3
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_presence_penalty: float = 0.0
    llm_repetition_penalty: float = 1.0
    llm_stream: bool = False
    llm_enable_thinking: bool = False
    llm_enabled_fail_open: bool = True
    llm_emit_raw_debug: bool = False
    llm_call_budget_per_round: int = 1
    llm_trigger_on_no_supported_path: bool = True
    llm_trigger_on_repeated_failures: bool = True
    llm_trigger_on_contradictions: bool = True
    llm_trigger_on_tied_deterministic: bool = True
    llm_trigger_on_graph_ambiguity: bool = True
    llm_confidence_cap: float = 0.45
    llm_prompt_max_nodes: int = 16
    llm_prompt_max_edges: int = 24
    llm_prompt_max_paths: int = 5
    llm_prompt_max_contradictions: int = 5
    llm_prompt_max_exit_attempts: int = 5
    llm_prompt_max_pattern_relations: int = 5
    llm_prompt_max_allowed_node_ids: int = 24
    llm_prompt_max_chars: int = 9000
    llm_prompt_max_approx_tokens: int = 2200
    llm_prompt_trim_strategy: str = "optional_then_limits"
    observed_graph_priority_weight: float = 1.0
    deterministic_priority_weight: float = 0.8
    validated_llm_priority_weight: float = 0.55
    unvalidated_llm_priority_weight: float = 0.2
    llm_mode_hypothesis_generator: dict[str, Any] = None
    llm_mode_ambiguity_resolver: dict[str, Any] = None
    llm_mode_experiment_suggester: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.llm_mode_hypothesis_generator is None:
            object.__setattr__(
                self,
                "llm_mode_hypothesis_generator",
                {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                    "max_output_tokens": 1200,
                },
            )
        if self.llm_mode_ambiguity_resolver is None:
            object.__setattr__(
                self,
                "llm_mode_ambiguity_resolver",
                {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 20,
                    "presence_penalty": 0.5,
                    "max_output_tokens": 1500,
                },
            )
        if self.llm_mode_experiment_suggester is None:
            object.__setattr__(
                self,
                "llm_mode_experiment_suggester",
                {
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 0.2,
                    "max_output_tokens": 1000,
                },
            )
