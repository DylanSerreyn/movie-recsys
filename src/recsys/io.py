"""
io.py — handles loading and inspecting MovieLens data
"""

import pandas as pd
from pathlib import Path

# Locate the data directory
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ml-32m"

def load_movies():
    """Load movies.csv"""
    return pd.read_csv(DATA_PATH / "movies.csv")

def load_ratings():
    """Load ratings.csv"""
    return pd.read_csv(DATA_PATH / "ratings.csv")

def load_tags():
    """Load tags.csv"""
    return pd.read_csv(DATA_PATH / "tags.csv")

def load_links():
    """Load links.csv"""
    return pd.read_csv(DATA_PATH / "links.csv")

def summarize():
    """Print basic info about all datasets"""
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    links = load_links()

    print("===== Dataset Summary =====")
    print(f"Movies:  {movies.shape}")
    print(f"Ratings: {ratings.shape}")
    print(f"Tags:    {tags.shape}")
    print(f"Links:   {links.shape}\n")

    print("Sample movies:")
    print(movies.head())
    print("\nSample ratings:")
    print(ratings.head())
