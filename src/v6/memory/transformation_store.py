from __future__ import annotations

import json
import sqlite3

import numpy as np

from v6.delta.delta_extractor import Delta
from v6.transformation.transformation_clusterer import TransformationFamily


class TransformationStore:
    def __init__(self, connection: sqlite3.Connection, *, auto_commit: bool = True) -> None:
        self.connection = connection
        self.auto_commit = bool(auto_commit)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS deltas (
                id INTEGER PRIMARY KEY,
                changed_cells INTEGER NOT NULL,
                changed_positions TEXT NOT NULL,
                colors_added TEXT NOT NULL,
                colors_removed TEXT NOT NULL,
                centroid_before_x REAL NOT NULL,
                centroid_before_y REAL NOT NULL,
                centroid_after_x REAL NOT NULL,
                centroid_after_y REAL NOT NULL,
                dx REAL NOT NULL,
                dy REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transformation_families (
                id INTEGER PRIMARY KEY,
                centroid_vector TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                member_delta_ids TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def next_delta_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM deltas").fetchone()
        return int(row[0])

    def add_delta(self, delta: Delta) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO deltas (
                id,
                changed_cells,
                changed_positions,
                colors_added,
                colors_removed,
                centroid_before_x,
                centroid_before_y,
                centroid_after_x,
                centroid_after_y,
                dx,
                dy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(delta.id),
                int(delta.changed_cells),
                json.dumps(delta.changed_positions),
                json.dumps(delta.colors_added),
                json.dumps(delta.colors_removed),
                float(delta.centroid_before_x),
                float(delta.centroid_before_y),
                float(delta.centroid_after_x),
                float(delta.centroid_after_y),
                float(delta.dx),
                float(delta.dy),
            ),
        )
        if self.auto_commit:
            self.connection.commit()

    def replace_families(self, families: dict[int, TransformationFamily]) -> None:
        self.connection.execute("DELETE FROM transformation_families")
        self.connection.executemany(
            """
            INSERT INTO transformation_families (
                id,
                centroid_vector,
                support_count,
                member_delta_ids
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    int(family.id),
                    json.dumps(np.asarray(family.centroid_vector, dtype=float).tolist()),
                    int(family.support_count),
                    json.dumps(family.member_delta_ids),
                )
                for family in families.values()
            ],
        )
        if self.auto_commit:
            self.connection.commit()

    def all_deltas(self) -> list[Delta]:
        rows = self.connection.execute(
            """
            SELECT id, changed_cells, changed_positions, colors_added, colors_removed,
                   centroid_before_x, centroid_before_y, centroid_after_x, centroid_after_y, dx, dy
            FROM deltas
            ORDER BY id
            """
        ).fetchall()
        return [
            Delta(
                id=int(row[0]),
                changed_cells=int(row[1]),
                changed_positions=[tuple(item) for item in json.loads(row[2])],
                colors_added=[int(value) for value in json.loads(row[3])],
                colors_removed=[int(value) for value in json.loads(row[4])],
                centroid_before_x=float(row[5]),
                centroid_before_y=float(row[6]),
                centroid_after_x=float(row[7]),
                centroid_after_y=float(row[8]),
                dx=float(row[9]),
                dy=float(row[10]),
            )
            for row in rows
        ]

    def all_families(self) -> list[TransformationFamily]:
        rows = self.connection.execute(
            """
            SELECT id, centroid_vector, support_count, member_delta_ids
            FROM transformation_families
            ORDER BY id
            """
        ).fetchall()
        return [
            TransformationFamily(
                id=int(row[0]),
                centroid_vector=np.asarray(json.loads(row[1]), dtype=float),
                support_count=int(row[2]),
                member_delta_ids=[int(value) for value in json.loads(row[3])],
            )
            for row in rows
        ]
