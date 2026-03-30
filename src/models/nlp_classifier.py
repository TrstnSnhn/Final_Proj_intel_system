from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class TfidfLogRegModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


def train_tfidf_logreg(texts: List[str], labels: List[str]) -> TfidfLogRegModel:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X, labels)
    return TfidfLogRegModel(vectorizer=vec, classifier=clf)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 64, num_filters: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)
