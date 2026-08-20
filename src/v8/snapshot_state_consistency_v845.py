from __future__ import annotations

import threading

_INSTALLED = False


def install_snapshot_state_consistency_v845() -> None:
    """Serialize persisted peer-state mutation with auxiliary snapshot capture."""
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import dedicated_lifecycle_v813 as lifecycle_module
    from v8 import peers_v82

    cls = peers_v82.V82DevelopmentalPeerSupervisor

    base_init = cls.__init__
    base_run_once = cls.run_once
    base_state_dict = cls.state_dict
    base_load_state = cls.load_state
    base_record_strategy_statistics = cls.record_strategy_statistics
    base_record_transfer_trial = cls.record_transfer_trial
    base_record_preference_probe = cls.record_preference_probe
    base_record_replanning_trial = cls.record_replanning_trial
    base_lifecycle_iteration = lifecycle_module._run_lifecycle_iteration

    def supervisor_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self._v845_state_lock = threading.RLock()

    def run_once(self):
        with self._v845_state_lock:
            return base_run_once(self)

    def state_dict(self):
        # The v8.13 fine-grained locks protect callers that invoke _fresh/_event_id
        # outside the normal peer/lifecycle wrappers. Acquire them while the shared
        # state lock is held so _seen/_sequence cannot change during serialization.
        with self._v845_state_lock:
            seen_lock = getattr(self, "_v813_seen_lock", None)
            event_lock = getattr(self, "_v813_event_lock", None)
            if seen_lock is None or event_lock is None:
                return base_state_dict(self)
            with seen_lock:
                with event_lock:
                    return base_state_dict(self)

    def load_state(self, state):
        with self._v845_state_lock:
            return base_load_state(self, state)

    def record_strategy_statistics(self, *args, **kwargs):
        with self._v845_state_lock:
            return base_record_strategy_statistics(self, *args, **kwargs)

    def record_transfer_trial(self, *args, **kwargs):
        with self._v845_state_lock:
            return base_record_transfer_trial(self, *args, **kwargs)

    def record_preference_probe(self, *args, **kwargs):
        with self._v845_state_lock:
            return base_record_preference_probe(self, *args, **kwargs)

    def record_replanning_trial(self, *args, **kwargs):
        with self._v845_state_lock:
            return base_record_replanning_trial(self, *args, **kwargs)

    def lifecycle_iteration(supervisor):
        with supervisor._v845_state_lock:
            return base_lifecycle_iteration(supervisor)

    cls.__init__ = supervisor_init
    cls.run_once = run_once
    cls.state_dict = state_dict
    cls.load_state = load_state
    cls.record_strategy_statistics = record_strategy_statistics
    cls.record_transfer_trial = record_transfer_trial
    cls.record_preference_probe = record_preference_probe
    cls.record_replanning_trial = record_replanning_trial
    lifecycle_module._run_lifecycle_iteration = lifecycle_iteration

    _INSTALLED = True
