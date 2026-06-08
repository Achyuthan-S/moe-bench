"""Tests for bootstrap routing CIs."""

from moe_bench.routing_uncertainty import bootstrap_routing_cis


def test_bootstrap_returns_intervals():
    per_layer = [
        {"mean_entropy": 3.5, "max_expert_util": 0.3, "collapsed": False, "num_tokens": 100},
        {"mean_entropy": 2.0, "max_expert_util": 0.5, "collapsed": True, "num_tokens": 100},
    ]
    out = bootstrap_routing_cis(per_layer, n_boot=500, seed=0)
    assert out["bootstrap_layers"] == 2
    assert len(out["mean_entropy_ci_95"]) == 2
    assert len(out["collapse_rate_ci_95"]) == 2
    assert out["mean_entropy_ci_95"][0] <= 3.5 <= out["mean_entropy_ci_95"][1]
