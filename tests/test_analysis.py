import pandas as pd

from moe_bench.analysis import recommend_collapse_threshold, sweep_collapse_thresholds


def test_threshold_sweep():
    df = pd.DataFrame(
        {
            "layer": [0, 1],
            "max_expert_util": [0.5, 0.2],
            "expert_0": [0.5, 0.1],
            "expert_1": [0.5, 0.9],
        }
    )
    sweep = sweep_collapse_thresholds(df, thresholds=[0.3, 0.4, 0.5])
    assert len(sweep) == 3
    assert recommend_collapse_threshold(sweep) in [0.3, 0.4, 0.5]
