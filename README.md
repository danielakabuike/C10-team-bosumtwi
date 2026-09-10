# C10-Team-Bosumtwi — Optimizing RAG Document Retrieval for Agronomic Advice

## Problem
Smallholder farmers and extension workers in West Africa often describe crop problems in informal, local terms, while extension documents use technical language. This vocabulary mismatch causes existing retrieval tools to rank keyword-similar but irrelevant documents above truly useful guidance. Our system improves the evidence-retrieval layer for a RAG-based agronomic advisory tool, using a hybrid lexical + dense + cross-encoder ranking pipeline.

## Dataset
We use the competition corpus located in `data/`:

- `documents.csv` — 695 agronomic extension documents with title, text, source, crop, country, origin and license.
- `train_queries.csv` — 308 farmer-style training queries with positive document hints.
- `qrels_train.csv` — 4,194 graded query-document relevance judgments (0 irrelevant, 3 highly relevant).
- `test_queries.csv` — 200 test queries without labels.
- `sample_submission.csv` & `baseline_submission.csv` — expected format and TF-IDF baseline.

The corpus blends synthetic CC0 documents with LLM-grounded rewrites of open-access CC-BY sources. No new data was collected from farmers; privacy is preserved by relying solely on the provided corpus.

## Training / Inference Pipeline
`scripts/retrieval_pipeline.py` implements:

1. **Preprocessing** — lowercasing, stripping non-alphanumeric characters, stopword removal and Porter stemming to build a BM25 token index; title, crop, country and body are concatenated into a single document field.
2. **BM25 lexical retrieval** (`rank_bm25`) over the cleaned/stemmed corpus.
3. **Dense retrieval** with `all-MiniLM-L12-v2` (fallback to `L6-v2`) encoding both queries and documents; cosine similarity returns the top-100 candidates.
4. **Reciprocal Rank Fusion (RRF)** to combine BM25 and dense rankings with tunable weights.
5. **Cross-encoder reranking** with `cross-encoder/ms-marco-MiniLM-L-12-v2` (fallback to `L-6-v2`) on the fused top-100, producing the final top-5 documents per query.
6. **Hyperparameter search** over `rrf_k`, `bm25_weight` and `dense_weight` on the training set to maximize nDCG@5.

## Evaluation
- Metric: **nDCG@5** on graded relevance labels.
- We tune RRF parameters on `train_queries.csv` + `qrels_train.csv` and report the best training nDCG@5.
- The final `submission.csv` is generated for `test_queries.csv` in the format required by the competition.

## Reproduction
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the inference/tuning script:
   ```bash
   cd C10-team-bosumtwi
   python scripts/retrieval_pipeline.py
   ```
   The script auto-detects `data/` in the repo, `agricultural-extension-rag-smart-retrieval-for-farmers/` locally, or Kaggle input paths.
3. The generated `submission.csv` contains 1,000 rows (200 queries × 5 ranked documents).

Alternatively, open `notebooks/agricproject_V1.ipynb` in Jupyter/Kaggle and run all cells.

## Repository Structure
```
C10-team-bosumtwi/
├── README.md
├── requirements.txt
├── scripts/
│   └── retrieval_pipeline.py
├── notebooks/
│   └── agricproject_V1.ipynb
├── data/
│   ├── documents.csv
│   ├── train_queries.csv
│   ├── test_queries.csv
│   ├── qrels_train.csv
│   ├── sample_submission.csv
│   └── baseline_submission.csv
└── docs/
    ├── problem_statement.pdf
    ├── data_card.pdf
    ├── impact_statement_card.pdf
    └── stakeholder_engagement.pdf
```

## Appendix: Contributors
- Team Bosumtwi — Cohort 10, TRI AI Saturdays
- Project mentor: Samuel Taiwo,Oluwaseun Ajayi,Adnan Adetunji

## References
- `docs/problem_statement.pdf`
- `docs/data_card.pdf`
- `docs/impact_statement_card.pdf`
- `docs/stakeholder_engagement.pdf`
