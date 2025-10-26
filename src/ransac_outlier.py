"""
ransac_outlier.py
Adaptive RANSAC for robust outlier removal (IITM-level implementation).

API:
    AdaptiveRANSAC.fit(observed, ideal, model_fit_fn, model_predict_fn, ...)
        - observed: (N,2) image points
        - ideal: (N,2) corresponding model points on plane
        - model_fit_fn: callable(sampled_ideal, sampled_obs) -> params
        - model_predict_fn: callable(ideal_all, params) -> predicted_obs_all (N,2)
Returns:
    inlier_mask (N,) boolean, best_params (object)
"""

from typing import Callable, Tuple, Optional, Any
import numpy as np
import random

class AdaptiveRANSAC:
    def __init__(self,
                 sample_size: int = 4,
                 max_iters: int = 2000,
                 base_thresh: float = 2.5,
                 alpha: float = 0.5,
                 confidence: float = 0.99,
                 rng_seed: int = 42):
        """
        Args:
            sample_size: minimal points needed to fit the model
            max_iters: maximum RANSAC iterations
            base_thresh: base threshold multiplier (residuals threshold = base_thresh * std(residuals))
            alpha: penalty weight for residual sum in scoring
            confidence: desired probability to have found a good model
            rng_seed: random seed for reproducibility
        """
        self.sample_size = sample_size
        self.max_iters = max_iters
        self.base_thresh = base_thresh
        self.alpha = alpha
        self.confidence = confidence
        self.rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

    def fit(self,
            observed: np.ndarray,
            ideal: np.ndarray,
            model_fit_fn: Callable[[np.ndarray, np.ndarray], Any],
            model_predict_fn: Callable[[np.ndarray, Any], np.ndarray],
            verbose: bool = False
            ) -> Tuple[np.ndarray, Optional[Any], dict]:
        """
        Run adaptive RANSAC.

        Args:
            observed: Nx2 array of observed image points.
            ideal: Nx2 array of corresponding ideal model points (plane).
            model_fit_fn: fits model from minimal sample; returns params.
            model_predict_fn: predicts observed points from ideal points + params; returns Nx2 array.
            verbose: print iteration info when True.

        Returns:
            best_mask: boolean mask of inliers (N,)
            best_params: parameters returned by model_fit_fn
            info: dictionary with extra diagnostics
        """
        N = observed.shape[0]
        if N < max(1, self.sample_size):
            return np.zeros(N, dtype=bool), None, {"reason": "insufficient_points"}

        best_mask = np.zeros(N, dtype=bool)
        best_params = None
        best_score = -np.inf
        best_inlier_count = 0

        # Precomputed ordering of indices for deterministic sampling
        indices = list(range(N))

        for it in range(self.max_iters):
            # Draw sample without replacement deterministically via RNG
            sample_idx = self.rng.sample(indices, min(self.sample_size, N))
            sampled_obs = observed[sample_idx]
            sampled_ideal = ideal[sample_idx]

            # Fit candidate model
            try:
                params = model_fit_fn(sampled_ideal, sampled_obs)
            except Exception:
                continue

            # Predict all observed positions using candidate params
            try:
                preds = model_predict_fn(ideal, params)  # (N,2)
            except Exception:
                continue

            # Residuals (Euclidean)
            residuals = np.linalg.norm(preds - observed, axis=1)
            # Adaptive threshold based on robust scale: use median absolute deviation
            mad = np.median(np.abs(residuals - np.median(residuals))) + 1e-8
            robust_std = 1.4826 * mad  # convert MAD to std approx
            if robust_std <= 0:
                robust_std = np.std(residuals) + 1e-8
            thresh = max(1e-8, self.base_thresh * robust_std)

            inlier_mask = residuals <= thresh
            inlier_count = int(inlier_mask.sum())
            if inlier_count == 0:
                continue

            # Score: favor more inliers and smaller residuals
            score = inlier_count - self.alpha * float(residuals[inlier_mask].sum())

            # Update best
            if (score > best_score) or (score == best_score and inlier_count > best_inlier_count):
                best_score = score
                best_mask = inlier_mask.copy()
                best_params = params
                best_inlier_count = inlier_count

            # Early termination: if confident enough about inlier ratio
            w = inlier_count / float(N)
            # Probabilistic required iterations
            if w > 0:
                s = self.sample_size
                # avoid math domain issues
                try:
                    import math
                    num_required = math.log(1 - self.confidence) / math.log(1 - w**s)
                    # small safety cap
                    if num_required < it + 1:
                        if verbose:
                            print(f"[RANSAC] Early stop at iter {it}, inlier_ratio={w:.3f}, required={num_required:.1f}")
                        break
                except Exception:
                    pass

        info = {
            "best_score": float(best_score),
            "best_inlier_count": int(best_inlier_count),
            "total_points": int(N),
            "iterations_run": it+1
        }
        return best_mask, best_params, info


# Example usage (simple illustration; integrate in pipeline)
if __name__ == "__main__":
    def toy_fit(ideal_s, obs_s):
        # trivial fit: compute translation that best maps ideal->obs (least squares)
        # Solve obs = ideal + t  => t = mean(obs-ideal)
        t = obs_s.mean(axis=0) - ideal_s.mean(axis=0)
        return {"t": t}

    def toy_predict(ideal_all, params):
        return ideal_all + params["t"]

    # Synthetic data
    np.random.seed(0)
    ideal = np.random.randn(100,2) * 30 + 200
    true_t = np.array([5.0, -3.0])
    obs = ideal + true_t
    # Add outliers
    obs[::10] += np.array([200, -150])

    ransac = AdaptiveRANSAC(sample_size=4, max_iters=1000, base_thresh=2.5, alpha=0.2, rng_seed=1)
    mask, params, info = ransac.fit(obs, ideal, toy_fit, toy_predict, verbose=True)
    print("Inliers:", mask.sum(), "Params:", params, "Info:", info)
