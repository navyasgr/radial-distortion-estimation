import numpy as np
from src.ransac_outlier import AdaptiveRANSAC

def test_ransac_basic_translation():
    # synthetic ideal points
    np.random.seed(2)
    ideal = np.random.randn(60,2) * 5 + 100
    t = np.array([10.0, -7.0])
    obs = ideal + t
    # add gross outliers
    obs[::8] += np.array([200, -100])

    def fit_fn(ideal_s, obs_s):
        return {"t": obs_s.mean(axis=0) - ideal_s.mean(axis=0)}

    def predict_fn(ideal_all, params):
        return ideal_all + params["t"]

    r = AdaptiveRANSAC(sample_size=4, max_iters=500, base_thresh=3.0, alpha=0.1, rng_seed=123)
    mask, params, info = r.fit(obs, ideal, fit_fn, predict_fn)
    assert mask.sum() > 40  # majority inliers recovered
    assert "t" in params
