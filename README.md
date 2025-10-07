Movie Recommendation System (MovieLens 32M + Streaming Availability)
====================================================================

Overview
--------
This project builds a simple, fast, and explainable movie recommender using the
MovieLens 32M dataset and a hybrid ML approach:

1) Collaborative Filtering (CF): TruncatedSVD on a user–item ratings matrix
   with a bias model (global + user + item) and latent factors.

2) Content-based: cosine similarity over movie genres; user profile built from
   the user’s highly-rated movies.

3) Hybrid: z-score blend of CF and Content scores:
   hybrid_score = ALPHA * z(CF) + (1-ALPHA) * z(Content)


Key Features
------------
- Efficient CF model (TruncatedSVD) with bias-centering and train-time sampling
  for speed on the 32M ratings set.
- Content model using genre vectors and a user preference profile.
- Hybrid ranking with a single knob (ALPHA) to balance CF vs Content.
- Popularity filters (min number of ratings) to avoid obscure, noisy items.
- Simple evaluation script with RMSE and Precision@K / Recall@K on a time-based split.


Repository Layout
-----------------
.
├─ main.py                  # Run hybrid recommendations for a given user
├─ main_eval.py             # Evaluation: RMSE, Precision@K, Recall@K
├─ requirements.txt         # Python dependencies (install with pip)
├─ .gitignore               # Ignores secrets and large CSVs
├─ data/
│  └─ ml-32m/               # Put MovieLens CSVs here (movies.csv, ratings.csv, tags.csv, links.csv)
└─ src/
   └─ recsys/
      ├─ io.py              # CSV loading helpers
      ├─ cf_sklearn.py      # TruncatedSVD collaborative filtering (no surprise dependency)
      ├─ content.py         # Genre-based content model + title resolver
      ├─ eval.py            # Metrics and per-user evaluation helpers
      └─ (optional) cache.py# On-disk cache (if you enable it)


Prerequisites
-------------
- Python 3.10+ recommended
- Windows PowerShell / VS Code (project developed/tested this way)
- MovieLens 32M CSV files (movies.csv, ratings.csv, tags.csv, links.csv)

Setup (Windows / PowerShell)
----------------------------
1) Create and activate a virtual environment:
   > python -m venv venv
   > Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   > .\venv\Scripts\Activate.ps1

2) Install dependencies:
   > pip install -r requirements.txt

3) Put MovieLens CSVs here:
   data\ml-32m\movies.csv
   data\ml-32m\ratings.csv
   data\ml-32m\tags.csv
   data\ml-32m\links.csv


How to Run
----------
A) Get hybrid recommendations for a user:
   > python main.py

   Adjustable knobs at the top of main.py:
     USER_ID      : which MovieLens user to recommend for
     TOPK         : how many recommendations to show
     ALPHA        : CF vs Content blend (0…1), e.g. 0.5
     SAMPLE_SIZE  : number of ratings to sample for CF training (speed knob)
     MIN_RATINGS  : minimum rating count per movie (popularity floor)
     COUNTRY      : availability country code (e.g., "us")

   Output includes CF/Content/Hybrid scores,  and rating counts
 
B) Evaluate the model (single user example with time-based split):
   > python main_eval.py

   Prints:
     - Train/Test sizes for the user
     - RMSE on an internal holdout and a global test
     - Precision@K / Recall@K (relative to the user’s future “relevant” ratings)
     - Diagnostic ranges for CF/Content/Hybrid
     - Sample Top CF and Top Hybrid lists


How It Works (Short Technical Notes)
------------------------------------
1) CF (src/recsys/cf_sklearn.py)
   - Build a sparse user–item matrix from a (sampled) ratings DataFrame.
   - Compute biases:
       global_mean, user_mean - global, item_mean - global
   - Bias-center observed entries, then run TruncatedSVD (n_components=128 by default).
   - Predict:
       pred = global + bu[user] + bi[item] + dot(U[user], V[item])
   - Only recommend items that exist in the trained (sampled) model.
   - Use a popularity floor (MIN_RATINGS) and clamp predictions to [0.5, 5.0] for sanity.

2) Content (src/recsys/content.py)
   - One-hot encode genres; create a movie–genre matrix.
   - Build a user profile by averaging the genre vectors of the user’s liked movies (rating ≥ 4).
   - Score each candidate by cosine similarity to that user profile.
   - Small tie-break by closeness of release year to preferred years.

3) Hybrid (main.py)
   - Standardize CF and Content scores across the candidate set (z-scores).
   - Compute: ALPHA * z(CF) + (1-ALPHA) * z(Content).
   - Sort descending to get Top-K.


Notes, Tips, and Common Issues
------------------------------
- Virtual environment: keep it activated when running scripts.
- Large data: don’t commit the CSVs; they are ignored via .gitignore.
- If predictions look “too flat” or surface obscure items:
  - Increase MIN_RATINGS (e.g., 1000 or 2000) to focus on mainstream/popular titles.
  - Ensure your target USER_ID is force-included in the CF training sample
    (handled inside cf_sklearn.fit()).
- To make runs faster: lower SAMPLE_SIZE in main.py (e.g., 150_000).


Repro Commands (Windows)
------------------------
# create & activate venv
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# install
pip install -r requirements.txt

# run
python main.py
python main_eval.py


Attribution
-----------
- MovieLens dataset: GroupLens Research (https://grouplens.org/datasets/movielens/)
