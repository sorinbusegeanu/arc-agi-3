from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from v6.delta.delta_extractor import Delta
from v6.transformation.feature_extractor import delta_to_feature_vector


@dataclass(frozen=True)
class TransformationFamily:
    id: int
    centroid_vector: np.ndarray
    support_count: int
    member_delta_ids: list[int]


class TransformationClusterer:
    def __init__(self, min_cluster_size: int = 5, metric: str = "euclidean", recluster_every: int = 100) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.metric = str(metric)
        self.recluster_every = int(recluster_every)
        self.families: dict[int, TransformationFamily] = {}
        self.delta_to_family: dict[int, int] = {}
        self._last_cluster_size = 0
        self._next_family_id = 1
        self._signature_to_family_id: dict[tuple[int, int, int, int, int], int] = {}

    def maybe_recluster(self, deltas: Iterable[Delta], interaction_count: int) -> bool:
        if interaction_count <= 0:
            return False
        if interaction_count % self.recluster_every != 0:
            return False
        deltas_list = list(deltas)
        if len(deltas_list) < self.min_cluster_size:
            return False
        self.fit(deltas_list)
        return True

    def fit(self, deltas: Iterable[Delta]) -> dict[int, int]:
        deltas_list = list(deltas)
        self._last_cluster_size = len(deltas_list)
        self.families = {}
        self.delta_to_family = {}
        if len(deltas_list) < self.min_cluster_size:
            return {}

        feature_matrix = np.vstack([delta_to_feature_vector(delta) for delta in deltas_list])
        labels = _run_hdbscan(feature_matrix, min_cluster_size=self.min_cluster_size, metric=self.metric)
        new_families: dict[int, TransformationFamily] = {}
        new_delta_to_family: dict[int, int] = {}

        for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
            indexes = np.where(labels == label)[0]
            member_delta_ids = [int(deltas_list[index].id) for index in indexes]
            centroid_vector = np.mean(feature_matrix[indexes], axis=0)
            family_id = self._family_id_for_centroid(centroid_vector)
            existing_family = new_families.get(family_id)
            if existing_family is None:
                family = TransformationFamily(
                    id=family_id,
                    centroid_vector=centroid_vector,
                    support_count=len(member_delta_ids),
                    member_delta_ids=member_delta_ids,
                )
            else:
                support_count = existing_family.support_count + len(member_delta_ids)
                centroid_vector = (
                    existing_family.centroid_vector * existing_family.support_count
                    + centroid_vector * len(member_delta_ids)
                ) / support_count
                family = TransformationFamily(
                    id=family_id,
                    centroid_vector=centroid_vector,
                    support_count=support_count,
                    member_delta_ids=existing_family.member_delta_ids + member_delta_ids,
                )
            new_families[family.id] = family
            for delta_id in member_delta_ids:
                new_delta_to_family[delta_id] = family.id
        self.families = new_families
        self.delta_to_family = new_delta_to_family
        return dict(self.delta_to_family)

    def family_for_delta(self, delta_id: int) -> int | None:
        return self.delta_to_family.get(int(delta_id))

    def nearest_family(self, delta: Delta) -> int | None:
        if not self.families:
            return None
        vector = delta_to_feature_vector(delta)
        family_id, _distance = min(
            (
                (family.id, float(np.linalg.norm(vector - family.centroid_vector)))
                for family in self.families.values()
            ),
            key=lambda item: item[1],
        )
        return int(family_id)

    def _family_id_for_centroid(self, centroid_vector: np.ndarray) -> int:
        signature = _centroid_signature(centroid_vector)
        existing = self._signature_to_family_id.get(signature)
        if existing is not None:
            return int(existing)
        family_id = self._next_family_id
        self._next_family_id += 1
        self._signature_to_family_id[signature] = family_id
        return int(family_id)


def _centroid_signature(centroid_vector: np.ndarray) -> tuple[int, int, int, int, int]:
    changed_cells, dx, dy, colors_added, colors_removed = np.asarray(centroid_vector, dtype=float).tolist()
    return (
        int(round(changed_cells / 8.0)),
        int(round(dx / 2.0)),
        int(round(dy / 2.0)),
        int(round(colors_added)),
        int(round(colors_removed)),
    )


def _run_hdbscan(feature_matrix: np.ndarray, *, min_cluster_size: int, metric: str) -> np.ndarray:
    try:
        import hdbscan  # type: ignore[import-not-found]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, core_dist_n_jobs=1)
        return np.asarray(clusterer.fit_predict(feature_matrix), dtype=int)
    except ImportError:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, allow_single_cluster=True, n_jobs=1)
        return np.asarray(clusterer.fit_predict(feature_matrix), dtype=int)
