
from typing import List, Tuple
from rank_bm25 import BM25Okapi

def get_top_n_sim_text(query: str, documents: List[str],top_n = 3):
    #from:https://www.cnpython.com/pypi/rank-bm25
    tokenized_corpus = []
    for doc in documents:
        text = []
        for char in doc:
            text.append(char)
        tokenized_corpus.append(text)

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = [char for char in query]
    #doc_scores = bm25.get_scores(tokenized_query)  # array([0.        , 0.93729472, 0.        ])

    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=top_n)
    results = ["".join(res) for res in results]
    return results

def generate_prompt( question: str, relevant_chunks: List[str]):
    prompt = f'根据文档内容来回答问题，问题是"{question}"，文档内容如下：\n'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt

def strict_generate_prompt( question: str, relevant_chunks: List[str]):
    prompt = f'严格根据文档内容来回答问题，回答不允许编造成分要符合原文内容，问题是"{question}"，文档内容如下：\n'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt


import numpy as np
def compute_sim_score( v1: np.ndarray, v2: np.ndarray) -> float:
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

import torch
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

