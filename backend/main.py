# backend/main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import re

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, RepeatedKFold

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from naive_bayes import TFIDFNaiveBayes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Naive Bayes Ultra Exhaustivo",
    description="TF-IDF, n-gram (1..4), alpha [0.05..6], 10fold CV, etc.",
    version="4.0"
)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewText(BaseModel):
    text: str

nb_classifier = TFIDFNaiveBayes()
MODEL_PATH = "model/tfidf_nb_model.json"

stop_words = None
lemmatizer = None

# -----------------------------------------------------
# Generar n-gramas
# -----------------------------------------------------
def generate_ngrams(tokens, n):
    return [
        "_".join(tokens[i:i+n])
        for i in range(len(tokens)-n+1)
    ]

def generate_1_2_3_4grams(tokens):
    unis = generate_ngrams(tokens, 1)
    bis = generate_ngrams(tokens, 2)
    tris = generate_ngrams(tokens, 3)
    fours = generate_ngrams(tokens, 4)
    return unis + bis + tris + fours

# -----------------------------------------------------
# Preprocesado
# -----------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9áéíóúüñ]', ' ', text)
    tokens = text.split()

    filtered = []
    for t in tokens:
        if t not in stop_words:
            lemma = lemmatizer.lemmatize(t)
            filtered.append(lemma)

    # n-gramas de 1..4
    return generate_1_2_3_4grams(filtered)

# -----------------------------------------------------
# Filtrado global tokens
# -----------------------------------------------------
def filter_tokens_global(docs_tokens, min_count=3, max_prop=0.9):
    from collections import defaultdict
    dfreq = defaultdict(int)
    N = len(docs_tokens)
    for doc in docs_tokens:
        unique_t = set(doc)
        for t in unique_t:
            dfreq[t] += 1

    to_remove = set()
    for t, freq in dfreq.items():
        if freq < min_count:
            to_remove.add(t)
        if freq > max_prop * N:
            to_remove.add(t)

    new_docs = []
    for doc in docs_tokens:
        filtered = [tk for tk in doc if tk not in to_remove]
        new_docs.append(filtered)
    return new_docs

# -----------------------------------------------------
# TF-IDF manual
# -----------------------------------------------------
def build_tfidf(docs_tokens):
    from collections import Counter, defaultdict
    N = len(docs_tokens)
    dfreq = defaultdict(int)
    for toks in docs_tokens:
        for tk in set(toks):
            dfreq[tk] += 1

    idf = {}
    for tk, df in dfreq.items():
        idf[tk] = np.log((N+1) / (df+1)) + 1

    tfidf_docs = []
    for doc in docs_tokens:
        c = Counter(doc)
        length = len(doc)
        tfmap = {}
        for tk, freq in c.items():
            tf = freq / length
            tfmap[tk] = tf * idf[tk]
        tfidf_docs.append(tfmap)
    return tfidf_docs

# -----------------------------------------------------
# Cross Validate masivo
# -----------------------------------------------------
def crossval_alpha(df, alphas, folds=10, repeats=1):
    if repeats > 1:
        rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        indices = list(rkf.split(df))
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        indices = list(kf.split(df))

    X = df["tfidf"].values
    y = df["mapped_label"].values

    best_alpha = 1.0
    best_score = 0.0

    for alpha in alphas:
        scores = []
        for train_idx, val_idx in indices:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = TFIDFNaiveBayes(alpha=alpha)
            clf.train(X_train, y_train)

            preds = [clf.predict(x) for x in X_val]
            f1 = f1_score(y_val, preds, pos_label="genuina")
            scores.append(f1)
        avg = np.mean(scores)
        print(f"[alpha={alpha}] F1={avg:.4f}")
        if avg > best_score:
            best_score = avg
            best_alpha = alpha

    print(f"**Mejor alpha={best_alpha}, F1={best_score:.4f}")
    return best_alpha

# -----------------------------------------------------
# Entrenamiento
# -----------------------------------------------------
def train_model():
    data_path = os.path.join("data", "reviews.csv")
    if not os.path.exists(data_path):
        print("[ERROR] Falta reviews.csv en /data")
        return

    df = pd.read_csv(data_path)

    needed = {"label", "text_"}
    if not needed.issubset(df.columns):
        print("[ERROR] Faltan label, text_")
        return

    def map_label(lab):
        if lab == "CG":
            return "genuina"
        elif lab == "OR":
            return "falsa"
        return None

    df["mapped_label"] = df["label"].apply(map_label)
    df.dropna(subset=["mapped_label"], inplace=True)
    df.rename(columns={"text_": "review"}, inplace=True)
    # Generar tokens
    docs_tokens = []
    labs = []
    for rev, lab in zip(df["review"], df["mapped_label"]):
        tk = preprocess_text(rev)
        if len(tk) >= 3:   # filtrar reseñas super cortas
            docs_tokens.append(tk)
            labs.append(lab)

    # Filtrar tokens raros/frecuentes
    docs_tokens = filter_tokens_global(docs_tokens, min_count=3, max_prop=0.9)

    # Build tf-idf
    tfidf_list = build_tfidf(docs_tokens)
    final_df = pd.DataFrame({
        "tfidf": tfidf_list,
        "mapped_label": labs
    })

    alpha_values = np.arange(0.05, 6.05, 0.05)

    # CrossVal con 10 folds, repetido 1 o más veces si deseas
    best_alpha = crossval_alpha(final_df, alpha_values, folds=10, repeats=1)

    # Entrenar final
    train_split = final_df.sample(frac=0.8, random_state=42)
    test_split = final_df.drop(train_split.index)

    global nb_classifier
    nb_classifier = TFIDFNaiveBayes(alpha=best_alpha)
    nb_classifier.train(train_split["tfidf"].tolist(), train_split["mapped_label"].tolist())
    nb_classifier.save_model(MODEL_PATH)

    # Evaluar
    X_test = test_split["tfidf"].tolist()
    y_test = test_split["mapped_label"].tolist()

    preds = [nb_classifier.predict(x) for x in X_test]
    acc = (np.array(preds) == np.array(y_test)).mean() if len(y_test) > 0 else 0
    pre = precision_score(y_test, preds, pos_label="genuina")
    rec = recall_score(y_test, preds, pos_label="genuina")
    f1v = f1_score(y_test, preds, pos_label="genuina")
    cm = confusion_matrix(y_test, preds, labels=["falsa", "genuina"])

    print("***** Entrenamiento Final Completado *****")
    print(f"Best alpha={best_alpha}")
    print(f"Test size: {len(y_test)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1v:.4f}")
    print("Matriz de Confusión (falsa/genuina):")
    print(cm)

@app.get("/train")
def train_endpoint():
    train_model()
    return {"message": "Entrenamiento masivo completado. Revisa la consola."}

@app.on_event("startup")
def startup_event():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    global stop_words
    global lemmatizer
    stop_words = set(stopwords.words("english"))  # Ajusta si es español
    lemmatizer = WordNetLemmatizer()

    if os.path.exists(MODEL_PATH):
        nb_classifier.load_model(MODEL_PATH)
        print("Modelo TF-IDF NB cargado.")
    else:
        print("No hay modelo, usa /train")

@app.post("/predict")
def predict_review(data: ReviewText):

    from collections import Counter
    toks = preprocess_text(data.text)
    c = Counter(toks)
    length = len(toks) if len(toks) > 0 else 1
    tf_map = {}
    for tk, freq in c.items():
        tf_map[tk] = freq / length  # sin IDF => aproximación
    pred = nb_classifier.predict(tf_map)
    return {"prediction": pred}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
