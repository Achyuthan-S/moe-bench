"""Unit tests for routing metrics (no GPU required)."""

import numpy as np
import pandas as pd

from moe_bench.metrics import (
    collapse_rate,
    compute_layer_metrics,
    expert_utilisation,
    load_imbalance_coefficient,
    routing_entropy,
)


def test_uniform_routing_high_entropy():
    probs = np.ones((100, 8)) / 8
    ent = routing_entropy(probs)
    assert np.allclose(ent.mean(), np.log(8), atol=1e-5)


def test_collapsed_routing_low_entropy():
    probs = np.zeros((100, 8))
    probs[:, 0] = 1.0
    ent = routing_entropy(probs)
    assert ent.mean() < 0.01


def test_expert_utilisation_argmax():
    probs = np.zeros((100, 8))
    probs[:, 3] = 1.0
    util = expert_utilisation(probs)
    assert util[3] == 1.0
    assert util.sum() == 1.0


def test_load_imbalance_uniform():
    util = np.ones(8) / 8
    assert load_imbalance_coefficient(util) == 1.0


def test_collapse_rate():
    utils = {0: np.array([0.5] + [0.5 / 7] * 7), 1: np.array([0.45] + [0.55 / 7] * 7)}
    rate, layers = collapse_rate(utils, threshold=0.40)
    assert rate == 1.0
    assert layers == [0, 1]


def test_compute_layer_metrics_dataframe():
    weights = {
        0: [np.ones((50, 4)) / 4],
        1: [np.tile([1, 0, 0, 0], (50, 1)).astype(float)],
    }
    df = compute_layer_metrics(weights, collapse_threshold=0.40)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.loc[df["layer"] == 1, "collapsed"].iloc[0]
