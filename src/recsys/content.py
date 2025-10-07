"""
content.py — memory-efficient content-based recommender (genres) with title resolution & smart tie-breaks
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from src.recsys.io import load_movies


# ---------- Title helpers ----------

ARTICLES = {"the", "a", "an"}

def _extract_year_from_title(title: str):
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None

def _normalize_for_lookup(title: str):
    """
    Lowercase, strip, move leading article to end (e.g., 'The Matrix' -> 'matrix, the'),
    and keep year if provided.
    """
    title = title.strip()
    year = _extract_year_from_title(title)
    base = title
    if year:
        base = title[: title.rfind("(")].strip()
    words = base.split()
    if words and words[0].lower() in ARTICLES:
        base_norm = " ".join(words[1:] + [words[0].lower()])
    else:
        base_norm = base
    base_norm = base_norm.lower()
    return f"{base_norm} ({year})" if year else base_norm

def _build_normalized_title_index(movies: pd.DataFrame):
    norm_map = {}
    for idx, t in movies["title"].items():
        norm_map[_normalize_for_lookup(t)] = idx
    return norm_map

def _resolve_title_to_index(query: str, movies: pd.DataFrame):
    """Resolve a user-typed title to the dataset index with simple normalization + fallback fuzzy search."""
    norm_index = _build_normalized_title_index(movies)
    qnorm = _normalize_for_lookup(query)
    if qnorm in norm_index:
        return norm_index[qnorm]

    # Fallback: soft match by token overlap (no extra deps)
    qtokens = set(re.sub(r"[^\w\s]", " ", qnorm).split())
    best_idx, best_score = None, -1
    for norm_title, idx in norm_index.items():
        tokens = set(re.sub(r"[^\w\s]", " ", norm_title).split())
        # Jaccard-like score with light year preference
        inter = len(qtokens & tokens)
        union = len(qtokens | tokens) or 1
        score = inter / union
        # slight boost if years match
        qy, ty = _extract_year_from_title(qnorm), _extract_year_from_title(norm_title)
        if qy and ty and qy == ty:
            score += 0.05
        if score > best_score:
            best_idx, best_score = idx, score

    return best_idx


# ---------- Genre features & recommendation ----------

def _prepare_genre_matrix(movies: pd.DataFrame):
    """Convert 'genres' to multi-hot binary matrix."""
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genre_list"])
    return movies, genre_matrix, mlb.classes_

def recommend_similar(title: str, n: int = 10):
    """
    Recommend movies similar to a given title using genre cosine similarity.
    Tie-breaks among identical vectors are resolved by closeness in year, then title.
    """
    movies = load_movies().copy()
    movies, X, _ = _prepare_genre_matrix(movies)

    idx = _resolve_title_to_index(title, movies)
    if idx is None:
        print(f"Movie '{title}' not found.")
        return pd.DataFrame()

    # Normalize rows for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    Xn = X / norms

    target_vec = Xn[idx:idx+1]
    sims = cosine_similarity(target_vec, Xn).ravel()

    # Build result frame (exclude itself)
    df = movies.copy()
    df["similarity"] = sims
    target_year = _extract_year_from_title(movies.at[idx, "title"])
    df["year"] = df["title"].apply(_extract_year_from_title)
    df["year_diff"] = df["year"].apply(lambda y: abs((y or 0) - (target_year or 0)))

    df = df.drop(index=idx)

    # Sort: similarity desc, year_diff asc, title asc
    df = df.sort_values(by=["similarity", "year_diff", "title"], ascending=[False, True, True])
    recs = df.head(n)[["title", "genres", "similarity", "year_diff"]]

    print(f"\n===== Movies similar to '{movies.at[idx, 'title']}' =====")
    print(recs)
    return recs
def user_content_scores(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.Series:
    """
    Mean cosine similarity (by genres) to the movies this user rated >= 4.0.
    Returns a Series aligned to movies_df.index (same order).
    """
    movies = movies_df.copy()
    movies["genre_list"] = movies["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(movies["genre_list"]).astype(float)

    # normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    Xn = X / norms

    liked = ratings_df[(ratings_df["userId"] == user_id) & (ratings_df["rating"] >= 4.0)]["movieId"]
    liked_idx = movies.index[movies["movieId"].isin(liked)].tolist()
    if not liked_idx:
        return pd.Series(0.0, index=movies.index)

    sims = Xn @ Xn[liked_idx].T            # (num_movies, num_liked)
    mean_sims = sims.mean(axis=1)          # (num_movies,)
    return pd.Series(mean_sims, index=movies.index)