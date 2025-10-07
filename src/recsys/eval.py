"""
eval.py — time-based evaluation for CF and Hybrid recommenders.

Provides:
- time_split_per_user(): per-user chronological split (train older, test newer)
- EvalCF: trains CFSklearn directly from a given train DataFrame
- evaluate_user(): RMSE on global test, plus Precision@K / Recall@K for CF and Hybrid
  and coverage diagnostics (how many relevant test items are in the candidate set)
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.recsys.io import load_ratings, load_movies
from src.recsys.cf_sklearn import CFSklearn, SVDConfig
from src.recsys.content import user_content_scores


# --------------------------
# Time-based per-user split
# --------------------------

def time_split_per_user(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split each user's ratings into train (older) / test (newer).
    Users with < min_interactions go entirely to train.
    """
    assert {"userId", "movieId", "rating", "timestamp"}.issubset(ratings.columns)

    train_parts, test_parts = [], []
    for uid, grp in ratings.groupby("userId", sort=False):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)
        if n < min_interactions:
            train_parts.append(grp_sorted)
            continue
        split_idx = int(np.floor(n * (1 - test_ratio)))
        split_idx = max(1, min(split_idx, n - 1))
        train_parts.append(grp_sorted.iloc[:split_idx])
        test_parts.append(grp_sorted.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train_df, test_df


# ----------------------------------------
# CF that fits from a provided DataFrame
# ----------------------------------------

class EvalCF(CFSklearn):
    """Train CFSklearn on a supplied TRAIN DataFrame."""

    def fit_from_dataframe(self, ratings_df: pd.DataFrame, ensure_user_id: Optional[int] = None):
        # Build ID maps from TRAIN
        users = ratings_df["userId"].unique()
        movies = ratings_df["movieId"].unique()
        uid_map = {uid: i for i, uid in enumerate(users)}
        mid_map = {mid: i for i, mid in enumerate(movies)}

        rows = ratings_df["userId"].map(uid_map).values
        cols = ratings_df["movieId"].map(mid_map).values
        data = ratings_df["rating"].astype(np.float32).values
        R = csr_matrix((data, (rows, cols)), shape=(len(uid_map), len(mid_map)))

        self.uid_map = uid_map
        self.mid_map = mid_map
        self.inv_mid_map = {v: k for k, v in mid_map.items()}

        # Biases on TRAIN only
        self._compute_biases(ratings_df)

        # Bias-center R
        n_users, n_items = R.shape
        bu_arr = np.zeros(n_users, dtype=np.float32)
        if isinstance(self.bu, pd.Series):
            for uid, val in self.bu.items():
                ui = self.uid_map.get(uid)
                if ui is not None:
                    bu_arr[ui] = float(val)

        bi_arr = np.zeros(n_items, dtype=np.float32)
        if isinstance(self.bi, pd.Series):
            for mid, val in self.bi.items():
                mi = self.mid_map.get(mid)
                if mi is not None:
                    bi_arr[mi] = float(val)

        Rc = R.tocoo(copy=True)
        base = self.global_mean + bu_arr[Rc.row] + bi_arr[Rc.col]
        Rc.data = Rc.data - base
        Rc = Rc.tocsr()

        # Train SVD
        self.U = self.svd.fit_transform(Rc)
        VT = self.svd.components_
        self.V = VT.T
        self._fitted = True

        # Internal quick RMSE on a random split of TRAIN (debugging only)
        tr, te = train_test_split(ratings_df, test_size=0.1, random_state=self.cfg.random_state)
        rmse_internal = self._rmse(te)
        return float(rmse_internal), R.shape[0], R.shape[1]


# --------------------------
# Top-N metrics
# --------------------------

@dataclass
class TopNMetrics:
    precision_at_k: float
    recall_at_k: float
    hits: int
    k: int
    num_relevant: int


def precision_recall_at_k(
    recommended_movie_ids: list[int],
    test_relevant_movie_ids: set[int],
    k: int,
) -> TopNMetrics:
    rec_k = recommended_movie_ids[:k]
    hits = sum(1 for m in rec_k if m in test_relevant_movie_ids)
    num_relevant = len(test_relevant_movie_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / num_relevant if num_relevant > 0 else 0.0
    return TopNMetrics(precision, recall, hits, k, num_relevant)


# -----------------------------------
# Single-user evaluation entry point
# -----------------------------------

def evaluate_user(
    user_id: int,
    k: int = 10,
    alpha: float = 0.6,
    rel_thresh: float = 4.0,
    test_ratio: float = 0.2,
    min_interactions: int = 10,
    min_ratings_item: int = 200,
    svd_components: int = 128,
) -> dict:
    """
    Time-based per-user split, CF trained on TRAIN only, and metrics on TEST.
    Returns a dict of metrics and top lists + coverage diagnostics.
    """
    ratings_full = load_ratings()
    movies = load_movies()

    # Split
    train_df, test_df = time_split_per_user(
        ratings_full, test_ratio=test_ratio, min_interactions=min_interactions
    )

    u_train = train_df[train_df["userId"] == user_id]
    u_test = test_df[test_df["userId"] == user_id]
    if u_train.empty or u_test.empty:
        return {"ok": False, "reason": f"User {user_id} lacks enough train/test (train={len(u_train)}, test={len(u_test)})"}

    # Train CF on TRAIN
    model = EvalCF(cfg=SVDConfig(n_components=svd_components, random_state=42))
    rmse_train_internal, n_users, n_items = model.fit_from_dataframe(train_df, ensure_user_id=user_id)

    # Global RMSE on TEST (all users). Clamp preds to [0.5, 5.0]
    def _predict_clamped(uid, mid):
        p = model._predict_ui(int(uid), int(mid))
        return float(min(5.0, max(0.5, p)))

    preds = test_df.apply(lambda r: _predict_clamped(r["userId"], r["movieId"]), axis=1)
    rmse_test = float(np.sqrt(np.mean((preds - test_df["rating"]) ** 2)))

    # Candidates for THIS user = unseen in TRAIN and IN the CF model
    seen_train = set(u_train["movieId"].tolist())
    in_model_movie_ids = set(model.mid_map.keys()) if model.mid_map else set()
    candidates = movies[movies["movieId"].isin(in_model_movie_ids) & ~movies["movieId"].isin(seen_train)].copy()

    # Popularity filter from TRAIN counts
    counts_train = train_df.groupby("movieId")["rating"].count()
    candidates = candidates.merge(
        counts_train.rename("rating_count"), left_on="movieId", right_index=True, how="left"
    )
    candidates["rating_count"] = candidates["rating_count"].fillna(0).astype(int)
    candidates = candidates[candidates["rating_count"] >= min_ratings_item]

    # Coverage diagnostics: how many relevant test items survive into candidates?
    relevant_test = set(u_test.loc[u_test["rating"] >= rel_thresh, "movieId"].tolist())
    candidate_ids = set(candidates["movieId"].tolist())
    overlap_relevant_in_candidates = sorted(list(relevant_test & candidate_ids))

    # CF ranking
    candidates["cf"] = candidates["movieId"].apply(lambda mid: model._predict_ui(user_id, int(mid)))
    top_cf = candidates.sort_values("cf", ascending=False).head(k)

    # Hybrid ranking (content uses TRAIN likes)
    content_all = user_content_scores(user_id=user_id, ratings_df=train_df, movies_df=movies)
    candidates["content"] = content_all.loc[candidates.index].values

    scaler = StandardScaler()
    candidates[["cf_z", "content_z"]] = scaler.fit_transform(candidates[["cf", "content"]])
    candidates["hybrid"] = alpha * candidates["cf_z"] + (1 - alpha) * candidates["content_z"]
    top_hybrid = candidates.sort_values("hybrid", ascending=False).head(k)

    # Top-N metrics vs TEST relevants
    cf_metrics = precision_recall_at_k(top_cf["movieId"].tolist(), relevant_test, k)
    hy_metrics = precision_recall_at_k(top_hybrid["movieId"].tolist(), relevant_test, k)

    return {
        "ok": True,
        "user_id": user_id,
        "train_size": int(len(u_train)),
        "test_size": int(len(u_test)),
        "svd_components": svd_components,
        "rmse_train_internal": float(rmse_train_internal),
        "rmse_test_global": float(rmse_test),

        # coverage info
        "relevant_in_test": len(relevant_test),
        "relevant_in_candidates": len(overlap_relevant_in_candidates),
        "relevant_ids_in_candidates": overlap_relevant_in_candidates,

        # top-N metrics
        "cf_precision_at_k": cf_metrics.precision_at_k,
        "cf_recall_at_k": cf_metrics.recall_at_k,
        "hy_precision_at_k": hy_metrics.precision_at_k,
        "hy_recall_at_k": hy_metrics.recall_at_k,
        "k": k,

        # debug ranges + lists
        "ranges": {
            "cf_min": float(candidates["cf"].min()) if not candidates.empty else None,
            "cf_max": float(candidates["cf"].max()) if not candidates.empty else None,
            "content_min": float(candidates["content"].min()) if not candidates.empty else None,
            "content_max": float(candidates["content"].max()) if not candidates.empty else None,
            "hybrid_min": float(candidates["hybrid"].min()) if not candidates.empty else None,
            "hybrid_max": float(candidates["hybrid"].max()) if not candidates.empty else None,
        },
        "top_cf": top_cf[["movieId", "title", "cf", "rating_count"]].reset_index(drop=True),
        "top_hybrid": top_hybrid[["movieId", "title", "cf", "content", "hybrid", "rating_count"]].reset_index(drop=True),
    }
