"""Vertical grid and profile averaging utilities."""

from __future__ import annotations

import numpy as np


def build_layer48_grid() -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate(([0.0, 500.0], np.arange(600.0, 2001.0, 100.0), np.arange(2250.0, 10001.0, 250.0))).astype(np.float32)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    if (len(edges), len(centers)) != (49, 48):
        raise RuntimeError("The physical grid must contain 49 edges and 48 layers")
    return edges, centers


def layer_average(source_height: np.ndarray, source_value: np.ndarray, edges: np.ndarray) -> np.ndarray:
    source_height, source_value = np.asarray(source_height), np.asarray(source_value)
    values = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        interior = source_height[(source_height > lower) & (source_height < upper)]
        sample_height = np.concatenate(([lower], interior, [upper]))
        sample_value = np.interp(sample_height, source_height, source_value)
        values.append(float(np.trapezoid(sample_value, sample_height) / (upper - lower)))
    return np.asarray(values, dtype=np.float32)
