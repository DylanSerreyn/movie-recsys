"""
cf_sklearn.py — Collaborative filtering with TruncatedSVD (no Surprise dependency).

- Samples ratings for speed, but FORCE-INCLUDES the target user's rows
- Bias model: global + user_bias + item_bias
- Centers by bias before SVD
- Predicts: base_bias + dot(U_user, V_item)
- Recommends only items that exist in the trained model, with an optional popularity filter
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from src.recsys.io import load_ratings, load_movies


@dataclass
class SVDConfig:
    n_components: int = 128
    random_state: int = 42


class CFSklearn:
    def __init__(self, cfg: SVDConfig = SVDConfig()):
        self.cfg = cfg
        self.svd = TruncatedSVD(n_components=cfg.n_components, random_state=cfg.random_state)

        # Bias model (computed on the same sample as SVD)
        self.global_mean: float = 0.0
        self.bu: pd.Series | None = None
        self.bi: pd.Series | None = None

        # Latent factors
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None

        # ID mappings (MovieLens IDs -> matrix indices) for the TRAINED MODEL
        self.uid_map: dict[int, int] | None = None
        self.mid_map: dict[int, int] | None = None
        self.inv_mid_map: dict[int, int] | None = None

        self._fitted: bool = False

    # ---------- helpers ----------

    @staticmethod
    def _build_mappings(ratings_df: pd.DataFrame):
        users = ratings_df["userId"].unique()
        movies = ratings_df["movieId"].unique()
        uid_map = {uid: i for i, uid in enumerate(users)}
        mid_map = {mid: i for i, mid in enumerate(movies)}
        return uid_map, mid_map

    @staticmethod
    def _build_csr(ratings_df: pd.DataFrame, uid_map: dict[int, int], mid_map: dict[int, int]) -> csr_matrix:
        rows = ratings_df["userId"].map(uid_map).values
        cols = ratings_df["movieId"].map(mid_map).values
        data = ratings_df["rating"].astype(np.float32).values
        n_users = len(uid_map)
        n_items = len(mid_map)
        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        return R

    def _compute_biases(self, ratings_df: pd.DataFrame) -> None:
        self.global_mean = float(ratings_df["rating"].mean())
        self.bu = ratings_df.groupby("userId")["rating"].mean() - self.global_mean
        self.bi = ratings_df.groupby("movieId")["rating"].mean() - self.global_mean

    def _predict_ui(self, user_id: int, movie_id: int) -> float:
        if not self._fitted or self.U is None or self.V is None:
            raise RuntimeError("Model not fitted. Call fit().")

        bu = float(self.bu.get(user_id, 0.0)) if isinstance(self.bu, pd.Series) else 0.0
        bi = float(self.bi.get(movie_id, 0.0)) if isinstance(self.bi, pd.Series) else 0.0
        base = self.global_mean + bu + bi

        ui = self.uid_map.get(user_id) if self.uid_map else None
        mi = self.mid_map.get(movie_id) if self.mid_map else None
        if ui is None or mi is None:
            return base

        pu = self.U[ui]
        qi = self.V[mi]
        return float(base + np.dot(pu, qi))

    def _rmse(self, df: pd.DataFrame) -> float:
        preds = df.apply(lambda r: self._predict_ui(int(r["userId"]), int(r["movieId"])), axis=1)
        mse = np.mean((preds - df["rating"]) ** 2)
        return float(np.sqrt(mse))

    # ---------- public API ----------

    def fit(self, sample_size: int = 300_000, ensure_user_id: Optional[int] = None) -> Tuple[float, int, int]:
        """
        Train SVD on a sampled ratings set. If ensure_user_id is provided, force-include
        all rows for that user in the training sample so they are in-model.
        """
        ratings_full = load_ratings()

        # Deterministic sample
        if sample_size and len(ratings_full) > sample_size:
            ratings_sample = ratings_full.sample(sample_size, random_state=self.cfg.random_state)
        else:
            ratings_sample = ratings_full.copy()

        # Force-include target user
        if ensure_user_id is not None:
            user_rows = ratings_full[ratings_full["userId"] == ensure_user_id]
            if not user_rows.empty:
                ratings_sample = (
                    pd.concat([ratings_sample, user_rows], ignore_index=True)
                      .drop_duplicates(subset=["userId", "movieId"], keep="first")
                )

        # Build model mappings/CSR from the sample
        uid_map, mid_map = self._build_mappings(ratings_sample)
        R = self._build_csr(ratings_sample, uid_map, mid_map)
        self.uid_map = uid_map
        self.mid_map = mid_map
        self.inv_mid_map = {v: k for k, v in mid_map.items()}

        # Biases on the same sample
        self._compute_biases(ratings_sample)

        # Bias-center
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

        # Fit SVD
        self.U = self.svd.fit_transform(Rc)
        VT = self.svd.components_
        self.V = VT.T

        self._fitted = True

        # Quick internal RMSE (from same sample; debug only)
        tr, te = train_test_split(ratings_sample, test_size=0.1, random_state=self.cfg.random_state)
        rmse = self._rmse(te)
        return rmse, n_users, n_items

    def recommend_for_user(self, user_id: int, k: int = 10, min_ratings: int = 500, clamp: bool = True) -> pd.DataFrame:
        """
        Recommend top-k unseen items that exist in the trained model,
        with a popularity floor (min_ratings) and optional clamping to [0.5, 5.0].
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit().")

        movies = load_movies()
        ratings = load_ratings()

        seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"].tolist())
        in_model_movie_ids = set(self.mid_map.keys()) if self.mid_map else set()
        candidates = movies[movies["movieId"].isin(in_model_movie_ids) & ~movies["movieId"].isin(seen)].copy()

        # Popularity filter
        counts = ratings.groupby("movieId")["rating"].count()
        candidates = candidates.merge(counts.rename("rating_count"), left_on="movieId", right_index=True, how="left")
        candidates["rating_count"] = candidates["rating_count"].fillna(0).astype(int)
        candidates = candidates[candidates["rating_count"] >= min_ratings]

        def _pred(mid: int) -> float:
            p = self._predict_ui(user_id, int(mid))
            return float(min(5.0, max(0.5, p))) if clamp else float(p)

        candidates["pred"] = candidates["movieId"].apply(_pred)
        recs = candidates.sort_values("pred", ascending=False).head(k)
        return recs[["movieId", "title", "genres", "pred", "rating_count"]].reset_index(drop=True)


def quick_train_and_recommend_sklearn(user_id: int = 1, k: int = 10, sample_size: int = 300_000, min_ratings: int = 500):
    model = CFSklearn()
    rmse, n_users, n_items = model.fit(sample_size=sample_size, ensure_user_id=user_id)
    print(f"TruncatedSVD CF — RMSE (holdout on sample): {rmse:.4f} | users={n_users} items={n_items}")
    recs = model.recommend_for_user(user_id=user_id, k=k, min_ratings=min_ratings, clamp=True)
    print(f"\n===== CF (sklearn) Top {k} for user {user_id} =====")
    for _, r in recs.iterrows():
        print(f"{r['title']:<50}  (pred {r['pred']:.3f})")
    return model, recs
