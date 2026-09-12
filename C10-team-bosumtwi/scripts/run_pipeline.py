"""
Agricultural Extension RAG: Dense + Sparse + LambdaMART pipeline.
Extracted from agricproject.ipynb and wrapped as a runnable script.

The script runs the feature extraction, fits the ranker, verifies all
row constraints, and writes the predictions to submission.csv.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data_dir ./Dataset --output submission.csv
"""

import argparse
import math
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRanker

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. Pure-Python BM25Okapi (matches rank-bm25, no extra dependency)
# ---------------------------------------------------------------------------
class BM25Okapi:
    """
    Pure Python/NumPy BM25Okapi implementation.
    Matches rank-bm25 formula and behavior exactly with zero dependencies.
    """

    def __init__(self, corpus, k1=1.2, b=0.5, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = (
            sum(len(x) for x in corpus) / self.corpus_size
            if self.corpus_size > 0
            else 0
        )
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = [len(x) for x in corpus]

        df = Counter()
        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] += 1

        # Calculate Okapi IDF with negative IDF smoothing
        negative_idfs = []
        for word, freq in df.items():
            idf_val = math.log(self.corpus_size - freq + 0.5) - math.log(
                freq + 0.5
            )
            self.idf[word] = idf_val
            if idf_val < 0:
                negative_idfs.append(word)

        self.average_idf = (
            sum(self.idf.values()) / len(self.idf) if self.idf else 0
        )
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size, dtype=np.float32)
        doc_len = np.array(self.doc_len, dtype=np.float32)

        for q_token in query:
            if q_token not in self.idf:
                continue
            q_idf = self.idf[q_token]
            q_freq = np.array(
                [doc.get(q_token, 0) for doc in self.doc_freqs],
                dtype=np.float32,
            )
            denom = q_freq + self.k1 * (
                1.0 - self.b + self.b * (doc_len / (self.avgdl + 1e-9))
            )
            score += q_idf * (q_freq * (self.k1 + 1.0)) / (denom + 1e-9)

        return score


# ---------------------------------------------------------------------------
# 1. CLI / path helpers
# ---------------------------------------------------------------------------
def find_data_dir(args_dir):
    """Resolve the competition data directory."""
    if args_dir:
        return args_dir

    # Check common local and Kaggle paths
    candidates = [
        "./Dataset",
        "./data",
        "./agricultural-extension-rag-smart-retrieval-for-farmers",
        "/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers",
        "/kaggle/input/agricultural-extension-rag-smart-retrieval-for-farmers",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "documents.csv")):
            return p

    # Fallback: walk Kaggle input directory
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "documents.csv" in files and "test_queries.csv" in files:
                return root

    raise FileNotFoundError(
        "Could not locate the competition data folder. "
        "Please pass --data_dir explicitly."
    )


def find_or_cache_model_dir(name, hf_id, cache_root):
    """Return a local path to a model. Download/cache if not present."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    # Check Kaggle input directories first
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if name.lower() in [d.lower() for d in dirs]:
                matched_path = os.path.join(root, name)
                print(f"Loaded local weights: {matched_path}")
                return matched_path

    # Check Kaggle working directory
    kaggle_working = f"/kaggle/working/{name}"
    if os.path.exists(kaggle_working):
        return kaggle_working

    # Check local cache
    local = os.path.join(cache_root, name)
    if os.path.exists(local):
        print(f"Using cached model: {local}")
        return local

    # Download and cache
    os.makedirs(cache_root, exist_ok=True)
    print(f"Downloading {name} from HuggingFace ({hf_id}) ...")
    if "reranker" in name.lower() or name.lower().startswith("bge-reranker"):
        CrossEncoder(hf_id).save(local)
    else:
        SentenceTransformer(hf_id).save(local)
    return local


# ---------------------------------------------------------------------------
# 2. Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the agricultural extension RAG pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to competition CSVs (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./model_cache",
        help="Where to cache downloaded models",
    )
    parser.add_argument(
        "--top_k_per_model",
        type=int,
        default=100,
        help="Candidate pool size per retriever (default: 100)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print("=" * 60)
    print("AGRICULTURAL EXTENSION RAG PIPELINE")
    print("Dense + Sparse + LambdaMART  |  Target: nDCG@5")
    print("=" * 60)
    # ------------------------------------------------------------------

    from sentence_transformers import SentenceTransformer, CrossEncoder
    from nltk.stem import WordNetLemmatizer
    import nltk

    # ── 1. SETUP & OFFLINE ASSET RESOLUTION ──────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on hardware device: {device}")

    # Offline support for NLTK tokenizers
    for p in ["/kaggle/input/nltk-data", "/usr/share/nltk_data", "/kaggle/working"]:
        if os.path.exists(p):
            nltk.data.path.append(p)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()

    DATA_DIR = find_data_dir(args.data_dir)
    print(f"Data directory: {DATA_DIR}")

    # Resolve model weights
    bge_path = find_or_cache_model_dir(
        "bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", args.cache_dir
    )
    e5_path = find_or_cache_model_dir(
        "e5-base-v2", "intfloat/e5-base-v2", args.cache_dir
    )
    minilm_path = find_or_cache_model_dir(
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        args.cache_dir,
    )
    reranker_path = find_or_cache_model_dir(
        "bge-reranker-large", "BAAI/bge-reranker-large", args.cache_dir
    )

    # ── 2. LOAD & ENRICH EXTENSION MATERIAL ──────────────────────────
    docs_df = pd.read_csv(os.path.join(DATA_DIR, "documents.csv"))
    train_queries_df = pd.read_csv(os.path.join(DATA_DIR, "train_queries.csv"))
    qrels_df = pd.read_csv(os.path.join(DATA_DIR, "qrels_train.csv"))
    test_queries_df = pd.read_csv(os.path.join(DATA_DIR, "test_queries.csv"))

    for col in ["title", "text", "crop", "country", "source"]:
        docs_df[col] = docs_df[col].fillna("").astype(str)

    docs_df["full_text"] = (
        "Crop: " + docs_df["crop"] + " | "
        + "Title: " + docs_df["title"] + " | "
        + "Country: " + docs_df["country"] + " | "
        + "Content: " + docs_df["text"]
    )

    # ── 3. LEXICAL INDEXING (BM25) ───────────────────────────────────
    def tokenize_for_bm25(text):
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [lemmatizer.lemmatize(t) for t in tokens if len(t) > 1]

    docs_df["bm25_tokens"] = docs_df["full_text"].apply(tokenize_for_bm25)
    bm25 = BM25Okapi(docs_df["bm25_tokens"].tolist(), k1=1.2, b=0.5)
    print(f"BM25 Index Built for {len(docs_df)} documents.")

    # ── 4. DENSE EMBEDDING RETRIEVAL ─────────────────────────────────
    bge_model = SentenceTransformer(bge_path, device=device)
    e5_model = SentenceTransformer(e5_path, device=device)
    minilm_model = SentenceTransformer(minilm_path, device=device)

    doc_texts = docs_df["full_text"].tolist()
    bge_doc_emb = bge_model.encode(
        doc_texts, normalize_embeddings=True, show_progress_bar=False
    )
    e5_doc_emb = e5_model.encode(
        ["passage: " + t for t in doc_texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    minilm_doc_emb = minilm_model.encode(
        doc_texts, normalize_embeddings=True, show_progress_bar=False
    )
    print("Dense representation indices computed.")

    # ── 5. FIRST-STAGE HYBRID CANDIDATE POOLING ──────────────────────
    def retrieve_hybrid_candidates(queries_df, top_k_per_model=100):
        candidates_records = []

        bge_q_emb = bge_model.encode(
            [
                "Represent this sentence for searching relevant passages: " + q
                for q in queries_df["query"]
            ],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        e5_q_emb = e5_model.encode(
            ["query: " + q for q in queries_df["query"]],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        minilm_q_emb = minilm_model.encode(
            queries_df["query"].tolist(),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        bge_sims = bge_q_emb @ bge_doc_emb.T
        e5_sims = e5_q_emb @ e5_doc_emb.T
        minilm_sims = minilm_q_emb @ minilm_doc_emb.T

        for i, row in enumerate(queries_df.itertuples()):
            qid = int(row.query_id)
            bm25_scores = bm25.get_scores(tokenize_for_bm25(row.query))

            top_bm = set(np.argsort(bm25_scores)[::-1][:top_k_per_model])
            top_bge = set(np.argsort(bge_sims[i])[::-1][:top_k_per_model])
            top_e5 = set(np.argsort(e5_sims[i])[::-1][:top_k_per_model])
            top_mini = set(np.argsort(minilm_sims[i])[::-1][:top_k_per_model])

            candidate_indices = set.union(top_bm, top_bge, top_e5, top_mini)

            for idx in candidate_indices:
                candidates_records.append(
                    {
                        "query_id": qid,
                        "document_id": docs_df.iloc[idx]["document_id"],
                        "bm25_score": float(bm25_scores[idx]),
                        "bge_score": float(bge_sims[i][idx]),
                        "e5_score": float(e5_sims[i][idx]),
                        "minilm_score": float(minilm_sims[i][idx]),
                    }
                )

        return pd.DataFrame(candidates_records)

    print("Extracting candidate pools for training and test...")
    train_candidates_df = retrieve_hybrid_candidates(
        train_queries_df, top_k_per_model=args.top_k_per_model
    )
    test_candidates_df = retrieve_hybrid_candidates(
        test_queries_df, top_k_per_model=args.top_k_per_model
    )

    # ── 6. CROSS-ENCODER SECOND-STAGE SCORING ────────────────────────
    reranker = CrossEncoder(reranker_path, device=device)
    doc_lookup = dict(zip(docs_df["document_id"], docs_df["full_text"]))

    train_q_lookup = dict(
        zip(train_queries_df["query_id"], train_queries_df["query"])
    )
    train_pairs = [
        [train_q_lookup[row.query_id], doc_lookup[row.document_id]]
        for row in train_candidates_df.itertuples()
    ]
    train_candidates_df["cross_encoder_score"] = reranker.predict(
        train_pairs, batch_size=16, show_progress_bar=True, convert_to_numpy=True
    )

    test_q_lookup = dict(
        zip(test_queries_df["query_id"], test_queries_df["query"])
    )
    test_pairs = [
        [test_q_lookup[row.query_id], doc_lookup[row.document_id]]
        for row in test_candidates_df.itertuples()
    ]
    test_candidates_df["cross_encoder_score"] = reranker.predict(
        test_pairs, batch_size=16, show_progress_bar=True, convert_to_numpy=True
    )

    # ── 7. FEATURE ENGINEERING & LABELED DATASETS ────────────────────
    def enrich_features(df, queries_df):
        merged = df.merge(queries_df[["query_id", "query"]], on="query_id")
        merged = merged.merge(
            docs_df[["document_id", "full_text", "crop"]], on="document_id"
        )

        merged["query_length"] = merged["query"].str.split().str.len()
        merged["document_length"] = merged["full_text"].str.split().str.len()
        merged["word_overlap"] = [
            len(set(q.lower().split()) & set(d.lower().split()))
            for q, d in zip(merged["query"], merged["full_text"])
        ]
        merged["crop_match"] = [
            int(c.lower() in q.lower()) if c else 0
            for c, q in zip(merged["crop"], merged["query"])
        ]
        return merged

    train_feat_df = enrich_features(train_candidates_df, train_queries_df)
    test_feat_df = enrich_features(test_candidates_df, test_queries_df)

    # Attach relevance labels; non-judged candidates become hard negatives (0)
    train_feat_df = train_feat_df.merge(
        qrels_df[["query_id", "document_id", "relevance"]],
        on=["query_id", "document_id"],
        how="left",
    )
    train_feat_df["relevance"] = train_feat_df["relevance"].fillna(0).astype(int)

    # ── 8. SUPERVISED LEARNING-TO-RANK (LAMBDAMART) ──────────────────
    feature_cols = [
        "bm25_score",
        "bge_score",
        "e5_score",
        "minilm_score",
        "cross_encoder_score",
        "query_length",
        "document_length",
        "word_overlap",
        "crop_match",
    ]

    train_feat_df = train_feat_df.sort_values("query_id").reset_index(drop=True)
    query_groups = train_feat_df.groupby("query_id", sort=False).size().tolist()

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_at=[5],
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    ranker.fit(
        train_feat_df[feature_cols],
        train_feat_df["relevance"],
        group=query_groups,
    )
    print("LambdaMART training concluded.")

    # ── COMPUTE OVERALL nDCG@5 ───────────────────────────────────────
    train_feat_df["prediction"] = ranker.predict(train_feat_df[feature_cols])

    def get_ndcg5(df):
        scores = []
        for _, group in df.groupby("query_id"):
            # DCG@5
            ranked = group.sort_values("prediction", ascending=False).head(5)
            rel = ranked["relevance"].to_numpy()
            dcg = np.sum(
                (2**rel - 1) / np.log2(np.arange(2, len(rel) + 2))
            )
            # IDCG@5
            ideal = group.sort_values("relevance", ascending=False).head(5)
            ideal_rel = ideal["relevance"].to_numpy()
            idcg = np.sum(
                (2**ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2))
            )
            scores.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(scores))

    print(f"Overall nDCG@5: {get_ndcg5(train_feat_df):.5f}")

    # ── 9. SUBMISSION GENERATION (LONG FORMAT: QueryId, DocumentId) ──
    test_feat_df["prediction"] = ranker.predict(test_feat_df[feature_cols])

    submission_rows = []
    test_query_order = test_queries_df["query_id"].tolist()

    for qid in test_query_order:
        q_candidates = test_feat_df[test_feat_df["query_id"] == qid]
        top_5 = (
            q_candidates.sort_values("prediction", ascending=False)
            .head(5)["document_id"]
            .tolist()
        )

        # Fallback to BM25 if fewer than 5 candidates were retrieved
        if len(top_5) < 5:
            q_text = test_queries_df.loc[
                test_queries_df["query_id"] == qid, "query"
            ].iloc[0]
            bm_fallback = np.argsort(
                bm25.get_scores(tokenize_for_bm25(q_text))
            )[::-1]
            for idx in bm_fallback:
                d_id = docs_df.iloc[idx]["document_id"]
                if d_id not in top_5:
                    top_5.append(d_id)
                if len(top_5) == 5:
                    break

        for doc_id in top_5:
            submission_rows.append({"QueryId": qid, "DocumentId": doc_id})

    submission_df = pd.DataFrame(submission_rows)

    # ── 10. STRICT VALIDATION & OUTPUT WRITE ─────────────────────────
    assert len(submission_df) == 1000, (
        f"Expected 1000 rows, got {len(submission_df)}"
    )
    assert list(submission_df.columns) == ["QueryId", "DocumentId"], (
        f"Incorrect columns: {submission_df.columns.tolist()}"
    )
    assert (submission_df.groupby("QueryId").size() == 5).all(), (
        "Every QueryId must have exactly 5 rows"
    )
    assert not submission_df.duplicated(
        subset=["QueryId", "DocumentId"]
    ).any(), "Duplicate (QueryId, DocumentId) pairs found"

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    submission_df.to_csv(args.output, index=False)

    print("\n" + "=" * 45)
    print("SUBMISSION VERIFIED & SAVED TO:")
    print(os.path.abspath(args.output))
    print("=" * 45)
    print(f"Total Rows: {len(submission_df)}")
    print(f"Columns: {list(submission_df.columns)}")
    print("\nFirst 10 Rows (Row order = Rank 1 to 5):")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()
