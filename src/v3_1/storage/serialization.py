from __future__ import annotations

import json

from v3_1.utils.serialization import to_plain_data


def dumps(payload) -> str:
    return json.dumps(to_plain_data(payload), sort_keys=True, indent=2)

