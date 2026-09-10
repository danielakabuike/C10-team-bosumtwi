import pandas as pd

print('documents:', pd.read_csv('data/documents.csv').shape)
print('train:', pd.read_csv('data/train_queries.csv').shape)
print('test:', pd.read_csv('data/test_queries.csv').shape)
print('qrels:', pd.read_csv('data/qrels_train.csv').shape)
print('sample:', pd.read_csv('data/sample_submission.csv').shape)
print('baseline:', pd.read_csv('data/baseline_submission.csv').shape)
