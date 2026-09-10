"""
Retrieval pipeline for the Agricultural Extension RAG competition.
Single-cell Kaggle notebook version.
"""
import importlib
import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

# install missing packages first so the imports below don't fail
for _mod, _pkg in {
    'rank_bm25': 'rank-bm25',
    'sentence_transformers': 'sentence-transformers',
    'nltk': 'nltk',
}.items():
    try:
        importlib.import_module(_mod)
    except ImportError:
        print(f"Installing {_pkg} ...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg])

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk.stem import PorterStemmer

# simple stopword list for BM25
default_stopwords = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'am', 's', 't', 'just', 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ma', 'about', 'above', 'after', 'again',
    'against', 'all', 'any', 'because', 'before', 'below', 'between', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
}

_stemmer = PorterStemmer()


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Tokenize for BM25: clean, remove stopwords, stem."""
    tokens = clean_text(text).split()
    return [_stemmer.stem(t) for t in tokens if t not in default_stopwords and len(t) > 1]


# find the data folder: local first, then Kaggle path, then auto-discover
def find_data_dir():
    # repo-local data folder
    if os.path.exists(os.path.join('data', 'documents.csv')):
        return os.path.abspath('data')
    # folder in the current working directory
    local_dir = "agricultural-extension-rag-smart-retrieval-for-farmers"
    # common Kaggle mount points
    kaggle_dirs = [
        "/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval",
        "/kaggle/input/agricultural-extension-rag-smart-retrieval-for-farmers",
    ]

    if os.path.exists(os.path.join(local_dir, 'documents.csv')):
        return os.path.abspath(local_dir)
    for kd in kaggle_dirs:
        if os.path.exists(os.path.join(kd, 'documents.csv')):
            return kd
    for root, _, files in os.walk('/kaggle/input'):
        if 'documents.csv' in files:
            return root
    return local_dir


data_dir = find_data_dir()
print(f"Using data directory: {data_dir}")

# confirm the files exist before reading
for _f in ['documents.csv', 'train_queries.csv', 'qrels_train.csv', 'test_queries.csv']:
    if not os.path.exists(os.path.join(data_dir, _f)):
        raise FileNotFoundError(f"Could not find {data_dir}/{_f}")

# load data
docs_df = pd.read_csv(os.path.join(data_dir, 'documents.csv'), dtype={'document_id': str})
train_queries_df = pd.read_csv(os.path.join(data_dir, 'train_queries.csv'))
qrels_df = pd.read_csv(os.path.join(data_dir, 'qrels_train.csv'), dtype={'query_id': str, 'document_id': str})
test_queries_df = pd.read_csv(os.path.join(data_dir, 'test_queries.csv'))

print(f"Documents: {len(docs_df)} | Train queries: {len(train_queries_df)} | "
      f"Test queries: {len(test_queries_df)} | Qrels: {len(qrels_df)}")

# document_id and query_id should be strings throughout
docs_df['document_id'] = docs_df['document_id'].astype(str)
train_queries_df['query_id'] = train_queries_df['query_id'].astype(str)
test_queries_df['query_id'] = test_queries_df['query_id'].astype(str)
qrels_df['query_id'] = qrels_df['query_id'].astype(str)
qrels_df['document_id'] = qrels_df['document_id'].astype(str)

# build a single text field from title, crop, country and body
docs_df['title'] = docs_df['title'].fillna('').astype(str)
docs_df['text'] = docs_df['text'].fillna('').astype(str)
docs_df['crop'] = docs_df.get('crop', pd.Series([''] * len(docs_df))).fillna('').astype(str)
docs_df['country'] = docs_df.get('country', pd.Series([''] * len(docs_df))).fillna('').astype(str)

docs_df['full_text'] = (
    docs_df['title'] + ' ' +
    docs_df['crop'] + ' ' +
    docs_df['country'] + ' ' +
    docs_df['text']
)
docs_df['full_text'] = docs_df['full_text'].apply(clean_text)

doc_text_lookup = dict(zip(docs_df['document_id'], docs_df['full_text']))

# BM25 index over the cleaned & stemmed documents
bm25 = BM25Okapi([tokenize(t) for t in docs_df['full_text']])

# set device and load encoders
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def load_sentence_transformer(names, device):
    for name in names:
        try:
            print(f"Loading sentence transformer: {name}")
            model = SentenceTransformer(name, device=device)
            model.max_seq_length = 256
            return model, name
        except Exception as exc:
            print(f"  Failed to load {name}: {exc}")
    raise RuntimeError("No sentence transformer could be loaded.")


def load_cross_encoder(names, device):
    for name in names:
        try:
            print(f"Loading cross-encoder: {name}")
            return CrossEncoder(name, device=device, max_length=512), name
        except Exception as exc:
            print(f"  Failed to load {name}: {exc}")
    raise RuntimeError("No cross-encoder could be loaded.")


bi_encoder, bi_encoder_name = load_sentence_transformer([
    'all-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'all-MiniLM-L6-v2',
    'sentence-transformers/all-MiniLM-L6-v2',
], device)

doc_embeddings = bi_encoder.encode(
    docs_df['full_text'].tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)

reranker, reranker_name = load_cross_encoder([
    'cross-encoder/ms-marco-MiniLM-L-12-v2',
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
], device)


# search helpers
def search_bm25(query, top_k=100):
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame([{
        'document_id': docs_df.iloc[i]['document_id'],
        'bm25_score': scores[i]
    } for i in top_indices])


def search_dense(query, top_k=100):
    query_emb = bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_emb, doc_embeddings)[0]
    topk = torch.topk(cosine_scores, top_k)
    return pd.DataFrame([{
        'document_id': docs_df.iloc[idx.item()]['document_id'],
        'dense_score': score.item()
    } for score, idx in zip(topk.values, topk.indices)])


def search_hybrid(query, top_k_candidates=100, rrf_k=60,
                  bm25_weight=1.0, dense_weight=1.0):
    """Combine BM25 and dense rankings with Reciprocal Rank Fusion."""
    bm25_df = search_bm25(query, top_k_candidates)
    dense_df = search_dense(query, top_k_candidates)

    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_df['document_id'], start=1):
        rrf_scores[doc_id] = bm25_weight / (rrf_k + rank)
    for rank, doc_id in enumerate(dense_df['document_id'], start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)

    hybrid = pd.DataFrame([
        {'document_id': doc_id, 'rrf_score': score}
        for doc_id, score in rrf_scores.items()
    ])
    return hybrid.sort_values('rrf_score', ascending=False).reset_index(drop=True)


def retrieve_and_rerank(query, top_candidates=100, top_final=5,
                          rrf_k=60, bm25_weight=1.0, dense_weight=1.0, batch_size=32):
    """Hybrid retrieval then cross-encoder reranking."""
    candidates = search_hybrid(
        query,
        top_k_candidates=top_candidates,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
    candidate_ids = candidates['document_id'].tolist()

    pairs = [[query, doc_text_lookup[doc_id]] for doc_id in candidate_ids]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    reranked = pd.DataFrame({
        'document_id': candidate_ids,
        'rerank_score': scores,
    }).sort_values('rerank_score', ascending=False).reset_index(drop=True)

    return reranked.head(top_final)


# evaluation using the competition metric: nDCG@5
def ndcg_at_k(scores, k=5):
    """Compute nDCG for a single query given ordered relevance scores."""
    scores = np.array(scores[:k])
    if len(scores) == 0:
        return 0.0
    dcg = np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))
    ideal = np.sort(scores)[::-1]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg(queries_df, qrels_df, k=5, **kwargs):
    """Average nDCG@k over the provided queries."""
    qrels = qrels_df.groupby('query_id').apply(
        lambda x: dict(zip(x['document_id'], x['relevance']))
    ).to_dict()

    total = 0.0
    for qid, qtext in queries_df[['query_id', 'query']].itertuples(index=False):
        top = retrieve_and_rerank(qtext, top_candidates=100, top_final=k, **kwargs)
        rels = qrels.get(qid, {})
        scores = [rels.get(doc_id, 0.0) for doc_id in top['document_id']]
        total += ndcg_at_k(scores, k=k)
    return total / max(len(queries_df), 1)


if len(train_queries_df) > 0 and len(qrels_df) > 0:
    def hybrid_mrr(queries_df, qrels_df, top_k=20, **hybrid_kwargs):
        qrels = qrels_df.groupby('query_id').apply(
            lambda x: dict(zip(x['document_id'], x['relevance']))
        ).to_dict()
        total = 0.0
        for qid, qtext in queries_df[['query_id', 'query']].itertuples(index=False):
            top = search_hybrid(qtext, top_k_candidates=top_k, **hybrid_kwargs)
            rels = qrels.get(qid, {})
            scores = [rels.get(doc_id, 0.0) for doc_id in top['document_id']]
            total += ndcg_at_k(scores, k=top_k)
        return total / max(len(queries_df), 1)

    print("\nTuning RRF weights on training queries...")
    best_score = -1.0
    best_params = (60, 1.0, 1.0)
    for rrf_k in (40, 60, 80):
        for bm25_w in (1.0, 1.5, 2.0):
            for dense_w in (1.0, 1.5, 2.0):
                score = hybrid_mrr(train_queries_df, qrels_df, top_k=5,
                                   rrf_k=rrf_k, bm25_weight=bm25_w,
                                   dense_weight=dense_w)
                print(f"  rrf_k={rrf_k:>2} bm25_w={bm25_w} dense_w={dense_w}  ->  nDCG@5={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params = (rrf_k, bm25_w, dense_w)

    rrf_k, bm25_w, dense_w = best_params
    print(f"\nBest hybrid params: rrf_k={rrf_k}, bm25_weight={bm25_w}, dense_weight={dense_w} (nDCG@5={best_score:.4f})")

    metrics = evaluate_ndcg(train_queries_df, qrels_df, k=5,
                            rrf_k=rrf_k, bm25_weight=bm25_w, dense_weight=dense_w)
    print(f"Full pipeline on train: nDCG@5={metrics:.4f}")
else:
    rrf_k, bm25_w, dense_w = 60, 1.0, 1.0

# generate submission
submission_rows = []
print(f"\nProcessing {len(test_queries_df)} test queries...")
for qid, qtext in test_queries_df[['query_id', 'query']].itertuples(index=False):
    top5 = retrieve_and_rerank(
        qtext,
        top_candidates=100,
        top_final=5,
        rrf_k=rrf_k,
        bm25_weight=bm25_w,
        dense_weight=dense_w,
    )
    for doc_id in top5['document_id']:
        submission_rows.append({'QueryId': qid, 'DocumentId': doc_id})

submission_df = pd.DataFrame(submission_rows)
output_path = 'submission.csv'
os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
submission_df.to_csv(output_path, index=False)

print(f"Saved {output_path}: {len(submission_df)} rows, "
      f"{submission_df['QueryId'].nunique()} unique queries.")
print(submission_df.head(10))
