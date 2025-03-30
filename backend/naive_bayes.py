# backend/naive_bayes.py

import math
import json
from collections import defaultdict

class TFIDFNaiveBayes:
    """
    Naïve Bayes implementado desde cero, con TF-IDF en lugar de conteos brutos.
    """

    def __init__(self, alpha=1.0):
        self.word_weights = {
            "falsa": defaultdict(float),
            "genuina": defaultdict(float)
        }
        self.total_weight = {
            "falsa": 0.0,
            "genuina": 0.0
        }
        self.docs_count = {
            "falsa": 0,
            "genuina": 0
        }
        self.priors = {
            "falsa": 0.0,
            "genuina": 0.0
        }
        self.vocabulary = set()
        self.alpha = alpha

    def train(self, tfidf_docs, labels):
        """
        tfidf_docs: lista de dicts {token->tfidf}, labels: lista de 'falsa'/'genuina'
        """
        for lab in labels:
            self.docs_count[lab] += 1
        total_docs = len(labels)

        for c in ["falsa","genuina"]:
            self.priors[c] = self.docs_count[c] / total_docs

        for doc_map, lab in zip(tfidf_docs, labels):
            for token, w in doc_map.items():
                self.word_weights[lab][token] += w
                self.total_weight[lab] += w
                self.vocabulary.add(token)

    def predict(self, tfidf_doc):
        score_falsa = math.log(self.priors["falsa"])
        score_genuina = math.log(self.priors["genuina"])

        vocab_size = len(self.vocabulary)
        for token, w in tfidf_doc.items():
            weight_falsa = self.word_weights["falsa"].get(token, 0.0)
            weight_genuina = self.word_weights["genuina"].get(token, 0.0)

            p_falsa = (weight_falsa + self.alpha) / (self.total_weight["falsa"] + self.alpha * vocab_size)
            p_genuina = (weight_genuina + self.alpha) / (self.total_weight["genuina"] + self.alpha * vocab_size)

            score_falsa += w * math.log(p_falsa)
            score_genuina += w * math.log(p_genuina)

        return "falsa" if score_falsa > score_genuina else "genuina"

    def save_model(self, path="model/tfidf_nb_model.json"):
        data = {
            "word_weights_falsa": dict(self.word_weights["falsa"]),
            "word_weights_genuina": dict(self.word_weights["genuina"]),
            "total_weight_falsa": self.total_weight["falsa"],
            "total_weight_genuina": self.total_weight["genuina"],
            "docs_count_falsa": self.docs_count["falsa"],
            "docs_count_genuina": self.docs_count["genuina"],
            "prior_falsa": self.priors["falsa"],
            "prior_genuina": self.priors["genuina"],
            "vocabulary": list(self.vocabulary),
            "alpha": self.alpha
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_model(self, path="model/tfidf_nb_model.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from collections import defaultdict
        self.word_weights = {
            "falsa": defaultdict(float, data["word_weights_falsa"]),
            "genuina": defaultdict(float, data["word_weights_genuina"])
        }
        self.total_weight["falsa"] = data["total_weight_falsa"]
        self.total_weight["genuina"] = data["total_weight_genuina"]
        self.docs_count["falsa"] = data["docs_count_falsa"]
        self.docs_count["genuina"] = data["docs_count_genuina"]
        self.priors["falsa"] = data["prior_falsa"]
        self.priors["genuina"] = data["prior_genuina"]
        self.vocabulary = set(data["vocabulary"])
        self.alpha = data["alpha"]
