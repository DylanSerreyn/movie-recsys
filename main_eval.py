# main_eval.py
from src.recsys.eval import evaluate_user

USER_ID = 1
K = 10
ALPHA = 0.6
REL_THRESH = 4.0
TEST_RATIO = 0.2
MIN_INTERACTIONS = 10
MIN_RATINGS_ITEM = 500
SVD_COMPONENTS = 128

if __name__ == "__main__":
    res = evaluate_user(
        user_id=USER_ID,
        k=K,
        alpha=ALPHA,
        rel_thresh=REL_THRESH,
        test_ratio=TEST_RATIO,
        min_interactions=MIN_INTERACTIONS,
        min_ratings_item=MIN_RATINGS_ITEM,
        svd_components=SVD_COMPONENTS,
    )

    if not res.get("ok", False):
        print("Eval skipped:", res.get("reason"))
    else:
        print(f"\n=== EVAL RESULTS for user {res['user_id']} ===")
        print(f"Train size: {res['train_size']} | Test size: {res['test_size']}")
        print(f"SVD components: {res['svd_components']}")
        print(f"RMSE (internal train split): {res['rmse_train_internal']:.4f}")
        print(f"RMSE (global TEST):          {res['rmse_test_global']:.4f}")

        print(f"\nCoverage (relevance threshold ≥ {REL_THRESH}):")
        print(f"  Relevant in TEST:        {res['relevant_in_test']}")
        print(f"  Relevant in CANDIDATES:  {res['relevant_in_candidates']}")
        print(f"  (IDs in candidates):     {res['relevant_ids_in_candidates']}")

        print(f"\nCF@{res['k']}:  precision={res['cf_precision_at_k']:.3f}  recall={res['cf_recall_at_k']:.3f}")
        print(f"HY@{res['k']}:  precision={res['hy_precision_at_k']:.3f}  recall={res['hy_recall_at_k']:.3f}")
        print("\nRanges:", res["ranges"])

        print("\nTop CF:")
        print(res["top_cf"].to_string(index=False))

        print("\nTop Hybrid:")
        print(res["top_hybrid"].to_string(index=False))

        # Quick guidance if coverage is zero
        if res['relevant_in_candidates'] == 0:
            print("\nTIP: No relevant test items are in the candidate set.")
            print("     Try one or more of:")
            print("       - Lower MIN_RATINGS_ITEM (e.g., 50)")
            print("       - Increase K (e.g., 20)")
            print("       - Adjust ALPHA (e.g., 0.5) to lean more on content")
