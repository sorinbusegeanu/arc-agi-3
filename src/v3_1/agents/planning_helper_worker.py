from __future__ import annotations

import ray

from v3_1.planning.helper_modes import run_helper_mode


@ray.remote
class PlanningHelperWorker:
    def run(self, request):
        return run_helper_mode(request)
