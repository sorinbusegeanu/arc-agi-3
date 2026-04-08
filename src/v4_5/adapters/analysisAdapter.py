from __future__ import annotations


class AnalysisAdapter:
    reused_modules = ("src/v4/analysis/*", "src/v4/memory/*", "src/v4/temporal/*", "src/v4/composition/*")

    def classify(self, expected: tuple[str, ...], observed: tuple[str, ...], terminal: bool = False, success: bool = False) -> str:
        if terminal and success:
            return "terminal_success"
        if terminal and not success:
            return "terminal_failure"
        if any(item in observed for item in ("unlock", "opened", "revealed")):
            return "unlock"
        if observed and observed == expected:
            return "progress"
        if observed and expected and observed != expected:
            return "partial_progress"
        if expected and not observed:
            return "contradiction"
        return "non_progress"
