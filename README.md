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


##  Training Pipeline
When testing early baselines, standard BM25, basic vector search, and even off-the-shelf cross-encoders by themselves struggled to hit high scores on nDCG@5. Farmer queries often use casual or local wording, while extension documents use technical agronomy terms. 

To fix this, we set up a two-stage pipeline: wide candidate retrieval across diverse models, followed by a gradient-boosted ranker.

[Farmer Query] │ ├── Standalone BM25 (Lemmatized) ──> Top 100 ├── BAAI/bge-small-en-v1.5 ──> Top 100 ├── intfloat/e5-base-v2 ──> Top 100 └── sentence-transformers/all-MiniLM-L6-v2 ──> Top 100 │ [Candidate Pooling] │ ├── Cross-Encoder Scoring (bge-reranker-large) ├── Metadata Matching (Crop Match, Word Overlap) └── Text Length Features │ [LightGBM LambdaMART Ranker] │ Top-5 Final Ranked Documents




### Pipeline Breakdown
1. **Lexical Retrieval (BM25)**: We wrote a pure NumPy/Python implementation of `BM25Okapi` ($k_1=1.2, b=0.5$) with NLTK lemmatization. It matches standard BM25 performance without needing external pip packages.
2. **Dense Embeddings**: We used three complementary models to capture semantic meaning:
   * `BAAI/bge-small-en-v1.5`
   * `intfloat/e5-base-v2`
   * `sentence-transformers/all-MiniLM-L6-v2`
3. **Candidate Pool**: We pulled the top 100 candidates from each dense model and BM25, merging them into a candidate pool for each query.
4. **Scoring & Features**: We ran candidates through `BAAI/bge-reranker-large` to get cross-attention scores. We then extracted 9 features per pair: the individual model scores, query/document lengths, token overlap, and a flag indicating whether the mentioned crop appears in the query.
5. **LambdaMART Reranking**: We trained a `LGBMRanker` using the `lambdarank` objective tuned directly for nDCG@5 (`learning_rate=0.05`, `n_estimators=150`, `num_leaves=31`).
6. **Safety Fallback**: If candidate pooling returns fewer than 5 docs for a query, the script automatically backfills the remaining slots with top BM25 matches to guarantee complete outputs.





## 3. Evaluation
We evaluated our ranking performance using Normalized Discounted Cumulative Gain at rank 5 (**nDCG@5**). 

* **Overall Train nDCG@5**: **0.96879**

Combining multiple dense encoders with BM25 gave the cross-encoder much better raw material to work with, and training LambdaMART directly on the ranking objective pulled the most relevant extension material into the top 5 slots.





## Reproduction
To run the full pipeline from scratch:

``bash
 1. Install required packages
pip install lightgbm sentence-transformers nltk torch pandas numpy

 2. Run the pipeline script
python scripts/run_pipeline.py
The script runs the feature extraction, fits the ranker, verifies all row constraints, and writes the predictions to submission.csv.

Alternatively, open `notebooks/agricproject.ipynb` in Jupyter/Kaggle and run all cells.

## Repository Structure

```
C10-team-bosumtwi/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_pipeline.py
├── notebooks/
│   └── agricproject.ipynb
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
- MEMBERS - AKABUIKE DANIEL (TEAM MEMEBER) , EMMANUEL DUROJAIYE
- Project mentor: Samuel Taiwo,Oluwaseun Ajayi,Adnan Adetunji

## References
- `docs/problem_statement.pdf`
- `docs/data_card.pdf`
- `docs/impact_statement_card.pdf`
- `docs/stakeholder_engagement.pdf`
