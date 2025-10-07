"""
baselines.py — simple baseline recommenders
"""

import pandas as pd
from src.recsys.io import load_ratings, load_movies

def most_popular(n: int = 10, min_ratings: int = 1000):
    """
    Recommend top-N movies by average rating and count.
    Args:
        n: number of movies to return
        min_ratings: minimum number of ratings to consider
    """
    ratings = load_ratings()
    movies = load_movies()

    # Compute average rating and rating count
    agg = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= min_ratings]  # filter unpopular movies

    # Weighted score: average boosted by log of rating count
    agg["score"] = agg["mean"] * 0.7 + (agg["count"].apply(lambda x: min(x, 10000)) / 10000) * 0.3

    # Join with movie titles
    result = pd.merge(agg, movies, on="movieId")
    result = result.sort_values("score", ascending=False).head(n)

    return result[["title", "mean", "count", "score", "genres"]]

def print_popular_examples():
    """Print top-N popular movies."""
    top = most_popular(n=10)
    print("===== Top 10 Most Popular Movies =====")
    for _, row in top.iterrows():
        print(f"{row['title']:<45}  ⭐ {row['mean']:.2f} ({int(row['count'])} ratings)")
