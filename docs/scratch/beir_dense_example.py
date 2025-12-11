from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import logging

logging.basicConfig(level=logging.INFO, handlers=[LoggingHandler()])

# 1) load data (BEIR format)
data_path = "/path/to/downloaded/beir_dataset"  # or use util.download_and_unzip(...)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# 2) create a dense encoder (any model in beir.retrieval.models)
encoder = models.SentenceBERT("msmarco-distilbert-base-v3")

# 3) wrap encoder in the exact dense search wrapper
dres = DRES(encoder, batch_size=32)

# 4) evaluation orchestration: choose score function ("cos_sim" or "dot")
retriever = EvaluateRetrieval(dres, score_function="cos_sim", k_values=[1, 3, 5, 10])

# 5) retrieve: returns dict[query_id] -> dict[doc_id] -> score
results = retriever.retrieve(corpus, queries)

# 6) evaluate: returns aggregated metrics (aligned with retriever.k_values)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

print("NDCG:", ndcg)
print("MAP:", _map)
print("Recall:", recall)
print("Precision:", precision)
print("MRR:", mrr)
