# src/recsys/availability.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# -----------------------------
# RapidAPI configuration
# -----------------------------
RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"

# Provider sometimes tweaks routes, so we try a couple variants.
PATHS_BY_IMDB = [
    "/shows/{imdb_id}",      # e.g., tt0068646
    "/shows/tt/{imdb_id}",   # some deployments prefix with /tt
]
PATHS_BY_TMDB = [
    "/shows/tmdb/{tmdb_id}", # e.g., movie/550
]

DEFAULT_COUNTRY = "us"
TYPE_LABEL = {"subscription": "sub", "free": "free", "buy": "buy", "rent": "rent", "addon": "addon"}

# -----------------------------
# Local data paths (your fixed path)
# -----------------------------
# Prefer MOVIELENS_LINKS from env (if set), otherwise use your provided absolute path.
DEFAULT_LINKS_PATH = Path(r"C:\Users\KingM\WPI\A25\CS4342\movie-recsys\data\ml-32m\links.csv")


# -----------------------------
# ENV / helpers
# -----------------------------

def _load_env_key() -> str:
    load_dotenv()
    key = os.getenv("RAPIDAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("RAPIDAPI_KEY not set. Create .env at repo root with RAPIDAPI_KEY=...")
    return key

def _links_csv_path() -> Path:
    load_dotenv()
    env_path = os.getenv("MOVIELENS_LINKS", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"MOVIELENS_LINKS='{env_path}' does not exist.")
        return p
    if not DEFAULT_LINKS_PATH.exists():
        raise FileNotFoundError(
            f"links.csv not found at {DEFAULT_LINKS_PATH}. "
            f"Set MOVIELENS_LINKS in .env to the full path of links.csv if it moved."
        )
    return DEFAULT_LINKS_PATH

def _load_links() -> pd.DataFrame:
    links = pd.read_csv(_links_csv_path())
    links["imdbId"] = pd.to_numeric(links.get("imdbId"), errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links.get("tmdbId"), errors="coerce").astype("Int64")
    return links

# -----------------------------
# API calls / normalization
# -----------------------------

def _rapidapi_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://{RAPIDAPI_HOST}{path}"
    headers = {"X-RapidAPI-Key": _load_env_key(), "X-RapidAPI-Host": RAPIDAPI_HOST}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    try:
        return r.json()
    except requests.JSONDecodeError:
        return {}

def _to_imdb_tt(imdb_int: Optional[int]) -> Optional[str]:
    if imdb_int is None or pd.isna(imdb_int):
        return None
    s = str(int(imdb_int))
    if len(s) < 7:
        s = s.zfill(7)
    return f"tt{s}"

def _tmdb_movie_path(tmdb_int: Optional[int]) -> Optional[str]:
    if tmdb_int is None or pd.isna(tmdb_int):
        return None
    return f"movie/{int(tmdb_int)}"

def _extract_offers(show_json: Dict[str, Any], country: str) -> List[str]:
    if not show_json:
        return []
    opts = (show_json.get("streamingOptions") or {}).get(country.lower()) \
           or (show_json.get("streamingOptions") or {}).get(country.upper()) \
           or []
    out, seen = [], set()
    for opt in opts:
        service = (opt.get("service") or {}).get("name") or opt.get("serviceId") or "Unknown"
        typ = str(opt.get("type") or "subscription").lower()
        label = TYPE_LABEL.get(typ, typ)
        addon = (opt.get("addon") or {}).get("name")
        if addon:
            label = f"{label}+{addon}"
        s = f"{service} ({label})"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# -----------------------------
# Public API
# -----------------------------

def get_availability_for_movieid(movie_id: int, country: str = DEFAULT_COUNTRY) -> List[str]:
    links = _load_links()
    row = links[links["movieId"] == movie_id]
    if row.empty:
        return []

    imdb_tt = _to_imdb_tt(row.iloc[0]["imdbId"])
    tmdb_path = _tmdb_movie_path(row.iloc[0]["tmdbId"])

    # Try IMDb variants first
    if imdb_tt:
        for path_tpl in PATHS_BY_IMDB:
            try:
                data = _rapidapi_get(path_tpl.format(imdb_id=imdb_tt), params={"country": country})
                offers = _extract_offers(data, country)
                if offers or data:
                    return offers  # if data but no offers, return empty (no providers in that country)
            except requests.HTTPError:
                continue

    # Fallback: TMDb variants
    if tmdb_path:
        for path_tpl in PATHS_BY_TMDB:
            try:
                data = _rapidapi_get(path_tpl.format(tmdb_id=tmdb_path), params={"country": country})
                offers = _extract_offers(data, country)
                if offers or data:
                    return offers
            except requests.HTTPError:
                continue

    return []

def annotate_with_availability(df: pd.DataFrame, country: str = DEFAULT_COUNTRY) -> pd.DataFrame:
    """Append 'availability' column to a DataFrame with 'movieId'."""
    def _lookup(mid: int) -> str:
        offers = get_availability_for_movieid(int(mid), country=country)
        return ", ".join(offers) if offers else "(no data)"
    out = df.copy()
    out["availability"] = out["movieId"].apply(_lookup)
    return out
