"""
start_server.py — Start `transformers serve` with VL model compatibility patches.

Problem
-------
`Qwen3VLConfig` and `Qwen3_5Config` are VL wrapper configs.  They store
num_key_value_heads / num_attention_heads / hidden_size / head_dim under
`text_config`, not at the top level.  The `PagedAttentionCache` in the
`transformers serve` continuous-batching path reads them directly off the
top-level config object, which raises AttributeError at generation time.

This script monkey-patches those config classes to add property descriptors
that proxy the missing attrs from `text_config`, then delegates to the
standard `transformers serve` entry-point so everything else is unchanged.

Usage
-----
    # Qwen3-VL  (Instruct)
    python src/llm_stack_agentic/start_server.py \\
        --model Qwen/Qwen3-VL-2B-Instruct --port 8000

    # Qwen3.5-VL (Instruct)
    python src/llm_stack_agentic/start_server.py \\
        --model Qwen/Qwen3.5-VL-2B-Instruct --port 8000

All flags are forwarded verbatim to `transformers serve`.
"""
from __future__ import annotations

import sys

# Attributes that PagedAttentionCache.__init__ reads directly off the config.
_PROXY_ATTRS = ("num_key_value_heads", "num_attention_heads", "hidden_size", "head_dim")


def _make_proxy(attr: str):
    """Return a property that reads *attr* from self.text_config."""
    def getter(self):
        tc = self.text_config
        val = getattr(tc, attr, None)
        if val is None and attr == "head_dim":
            # Fallback: head_dim = hidden_size / num_attention_heads
            val = tc.hidden_size // tc.num_attention_heads
        if val is None:
            raise AttributeError(
                f"'{type(self).__name__}.text_config' has no attribute '{attr}'"
            )
        return val
    getter.__name__ = attr
    return property(getter)


def patch_vl_configs() -> None:
    """Add top-level proxy properties to VL wrapper config classes."""
    import transformers

    for cls_name in ("Qwen3VLConfig", "Qwen3_5Config"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        patched = []
        for attr in _PROXY_ATTRS:
            # Only add if the class itself hasn't defined it already.
            if attr not in vars(cls):
                setattr(cls, attr, _make_proxy(attr))
                patched.append(attr)
        if patched:
            print(f"[start_server] Patched {cls_name}: {patched}", flush=True)


if __name__ == "__main__":
    patch_vl_configs()
    # Forward remaining argv to `transformers serve`.
    sys.argv = ["transformers", "serve"] + sys.argv[1:]
    import runpy
    runpy.run_module("transformers", run_name="__main__", alter_sys=True)
