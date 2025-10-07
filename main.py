# main.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.recsys.io import load_ratings, load_movies
from src.recsys.cf_sklearn import quick_train_and_recommend_sklearn
from src.recsys.content import user_content_scores
from src.recsys.availability import annotate_with_availability

# ----- knobs you can tweak -----
USER_ID      = 1
TOPK         = 10
ALPHA        = 0.6
SAMPLE_SIZE  = 300_000
MIN_RATINGS  = 1000
COUNTRY      = "us"
# --------------------------------

def show_user_history(user_id: int, topk: int = 5):
    ratings = load_ratings()
    movies = load_movies()
    u = ratings[ratings["userId"] == user_id]
    if u.empty:
        print(f"No ratings found for user {user_id}")
        return
    u = u.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")

    print(f"\n===== User {user_id} — highest rated (top {topk}) =====")
    print(u.sort_values("rating", ascending=False)
            .head(topk)[["rating", "title", "genres"]]
            .to_string(index=False))

    print(f"\n===== User {user_id} — lowest rated (bottom {topk}) =====")
    print(u.sort_values("rating", ascending=True)
            .head(topk)[["rating", "title", "genres"]]
            .to_string(index=False))

if __name__ == "__main__":
    # 1) Inspect this user's real history
    show_user_history(USER_ID, topk=5)

    # 2) Train CF and show CF Top-N (now uses MIN_RATINGS internally + clamping)
    model, _ = quick_train_and_recommend_sklearn(user_id=USER_ID, k=TOPK, sample_size=SAMPLE_SIZE, min_ratings=MIN_RATINGS)

    # 3) Build a hybrid ranking
    ratings = load_ratings()
    movies  = load_movies()

    seen = set(ratings.loc[ratings["userId"] == USER_ID, "movieId"].tolist())
    candidates = movies[~movies["movieId"].isin(seen)].copy()

    # Restrict to in-model items
    in_model_movie_ids = set(model.mid_map.keys()) if model.mid_map else set()
    candidates = candidates[candidates["movieId"].isin(in_model_movie_ids)]

    # Popularity filter (same as CF preview)
    counts = ratings.groupby("movieId")["rating"].count()
    candidates = candidates.merge(counts.rename("rating_count"), left_on="movieId", right_index=True, how="left")
    candidates["rating_count"] = candidates["rating_count"].fillna(0).astype(int)
    candidates = candidates[candidates["rating_count"] >= MIN_RATINGS]

    # CF predictions (clamped for display sanity)
    candidates["cf"] = candidates["movieId"].apply(lambda mid: max(0.5, min(5.0, model._predict_ui(USER_ID, int(mid)))))

    # Content score (based on this user's history in full ratings)
    content_all = user_content_scores(user_id=USER_ID, ratings_df=ratings, movies_df=movies)
    candidates["content"] = content_all.loc[candidates.index].values

    # Blend
    scaler = StandardScaler()
    candidates[["cf_z", "content_z"]] = scaler.fit_transform(candidates[["cf", "content"]])
    candidates["hybrid"] = ALPHA * candidates["cf_z"] + (1 - ALPHA) * candidates["content_z"]

    # Rank
    top_hybrid = candidates.sort_values("hybrid", ascending=False).head(TOPK)

    # Append availability (does NOT change ranking)
    try:
        top_hybrid = annotate_with_availability(top_hybrid, country=COUNTRY)
    except Exception as e:
        print(f"(availability lookup skipped: {e})")

    # Debug ranges
    print("\nRanges:")
    print(f"  CF:      min={candidates['cf'].min():.4f}  max={candidates['cf'].max():.4f}")
    print(f"  Content: min={candidates['content'].min():.4f}  max={candidates['content'].max():.4f}")
    print(f"  Hybrid:  min={candidates['hybrid'].min():.4f}  max={candidates['hybrid'].max():.4f}")

    print(f"\n===== HYBRID Top {TOPK} for user {USER_ID} (alpha={ALPHA}, min_ratings={MIN_RATINGS}, country={COUNTRY}) =====")
    for _, r in top_hybrid.iterrows():
        print(
            f"{r['title']:<55}  "
            f"cf={r['cf']:.4f}  content={r['content']:.4f}  hybrid={r['hybrid']:.4f}  "
            f"(count={r['rating_count']})  {r.get('availability','')}"
        )
