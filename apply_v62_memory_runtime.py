from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path.cwd()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def inject_call_kwarg(text: str, call_marker: str, kwarg_line: str) -> str:
    start = text.find(call_marker)
    if start < 0:
        raise RuntimeError(f"missing call marker: {call_marker}")
    if kwarg_line.strip() in text[start : start + 2500]:
        return text
    open_idx = text.find("(", start)
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                line_start = text.rfind("\n", open_idx, i) + 1
                closing_prefix = text[line_start:i]
                base_indent = closing_prefix[: len(closing_prefix) - len(closing_prefix.lstrip())]
                insertion = f"{base_indent}    {kwarg_line.strip()}\n"
                return text[:line_start] + insertion + text[line_start:]
    raise RuntimeError(f"unbalanced call: {call_marker}")


def install_files() -> None:
    for rel in [
        Path("src/v6/memory/v62_runtime.py"),
        Path("src/v6/memory/migrations/v62.py"),
        Path("src/v6/tests/test_v62_memory_runtime.py"),
    ]:
        src = ROOT / rel
        dst = REPO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def patch_main() -> None:
    path = REPO / "src/v6/main.py"
    text = path.read_text(encoding="utf-8")
    original = text
    text = replace_once(
        text,
        "from v6.memory.query_engine import MemoryQueryEngine\n",
        "from v6.memory.query_engine import MemoryQueryEngine\nfrom v6.memory.v62_runtime import LearnedFutureOptionEstimator, V62MemoryController\n",
        "v62 import",
    )
    text = text.replace("    memory_query_enabled: bool = False\n", "    memory_query_enabled: bool = True\n", 1)
    text = text.replace("    memory_action_selection_enabled: bool = False\n", "    memory_action_selection_enabled: bool = True\n", 1)
    text = replace_once(
        text,
        "        self.future_option_estimator = FutureOptionEstimator()\n",
        "        self.future_option_estimator = LearnedFutureOptionEstimator(self.connection, fallback=FutureOptionEstimator())\n",
        "future option estimator",
    )
    query_block = """        self.memory_query = MemoryQueryEngine(\n            self.memory,\n            contingency_learner=self.contingency_learner,\n            graph=self.graph,\n        )\n"""
    controller_block = query_block + """        self.memory_controller = V62MemoryController(\n            self.memory,\n            contingency_learner=self.contingency_learner,\n            graph=self.graph,\n            query_engine=self.memory_query,\n            promotion_engine=self.promotion_engine,\n            context_head=self.context_contradictions,\n            carrier_head=self.carrier_tracker,\n            lifecycle_head=self.memory_lifecycle,\n            efficiency_head=self.efficiency_tracker,\n        )\n        self.memory_query = self.memory_controller.query_engine\n        self.promotion_engine = self.memory_controller.promotion_engine\n        self.context_contradictions = self.memory_controller.context_head\n        self.carrier_tracker = self.memory_controller.carrier_head\n        self.memory_lifecycle = self.memory_controller.lifecycle_head\n        self.efficiency_tracker = self.memory_controller.efficiency_head\n"""
    text = replace_once(text, query_block, controller_block, "controller integration")
    text = text.replace(
        "            self.memory_query = memory_query_engine\n",
        "            self.memory_controller.query_engine = memory_query_engine\n            self.memory_query = memory_query_engine\n",
        1,
    )
    text = text.replace("self.memory_query.rank_actions(", "self.memory_controller.choose_action_candidates(")
    text = text.replace(
        "self.memory_query.predict_family(context_signatures, action, record_query=False)",
        "self.memory_controller.predict(context_signatures, action, record_query=False)",
    )
    text = text.replace("self.promotion_engine.run_all(", "self.memory_controller.promote_candidates(")
    text = inject_call_kwarg(text, "compute_interaction_significance(", "weights=self.memory_controller.current_isf_weights(),")
    if text == original:
        raise RuntimeError("main.py was not changed")
    backup = path.with_suffix(".py.v61_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_migrations_init() -> None:
    path = REPO / "src/v6/memory/migrations/__init__.py"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    line = "from v6.memory.migrations.v62 import migrate_connection as migrate_v62_connection\n"
    if line not in text:
        path.write_text(text + ("\n" if text and not text.endswith("\n") else "") + line, encoding="utf-8")


def main() -> None:
    if not (REPO / "src/v6/main.py").exists():
        raise SystemExit("Run this script from the arc-agi-3 repository root")
    install_files()
    patch_main()
    patch_migrations_init()
    print("ARC-AGI3 v6.2 memory-runtime drop-in installed")


if __name__ == "__main__":
    main()
