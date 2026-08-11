from __future__ import annotations

from pathlib import Path

p = Path("src/v6/h07_h08_evidence_repairs.py")
text = p.read_text(encoding="utf-8")

if "import inspect\n" not in text:
    text = text.replace("import json\n", "import inspect\nimport json\n", 1)

old = '''        def build_functional(*args: Any, **kwargs: Any):\n            result = original_build(*args, **kwargs)\n            events, diagnostics, state = result\n'''
new = '''        def build_functional(*args: Any, **kwargs: Any):\n            signature = inspect.signature(original_build)\n            accepts_var_kwargs = any(\n                parameter.kind == inspect.Parameter.VAR_KEYWORD\n                for parameter in signature.parameters.values()\n            )\n            supported_kwargs = (\n                kwargs\n                if accepts_var_kwargs\n                else {\n                    key: value\n                    for key, value in kwargs.items()\n                    if key in signature.parameters\n                }\n            )\n            result = original_build(*args, **supported_kwargs)\n            events, diagnostics, state = result\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("expected H07/H08 build_functional wrapper not found")
    text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
