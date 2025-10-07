"""
preprocess.py — build user/movie ID maps and sparse rating matrix
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

from src.recsys.io import load_ratings

def build_mappings(ratings: pd.DataFrame):
    """Return user→index and movie→index dicts."""
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    uid_map = {uid: i for i, uid in enumerate(user_ids)}
    mid_map = {mid: i for i, mid in enumerate(movie_ids)}

    print(f"Unique users: {len(uid_map)} | Unique movies: {len(mid_map)}")
    return uid_map, mid_map

def build_sparse_matrix(ratings: pd.DataFrame, uid_map, mid_map):
    """Return CSR sparse user–movie matrix."""
    user_idx = ratings["userId"].map(uid_map)
    movie_idx = ratings["movieId"].map(mid_map)
    data = ratings["rating"].astype(np.float32)

    matrix = csr_matrix(
        (data, (user_idx, movie_idx)),
        shape=(len(uid_map), len(mid_map))
    )
    print(f"Sparse matrix shape: {matrix.shape}")
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Density: {density:.6f}")
    return matrix

def run_preprocessing(sample_size: int = 500000):
    """
    Load ratings, optionally sample for speed, build mappings and sparse matrix.
    """
    ratings = load_ratings()
    print(f"Loaded {len(ratings):,} ratings.")

    # For speed during development, take a random subset
    if sample_size and len(ratings) > sample_size:
        ratings = ratings.sample(sample_size, random_state=42)
        print(f"Sampled {len(ratings):,} ratings for development.")

    uid_map, mid_map = build_mappings(ratings)
    R = build_sparse_matrix(ratings, uid_map, mid_map)
    return R, uid_map, mid_map
