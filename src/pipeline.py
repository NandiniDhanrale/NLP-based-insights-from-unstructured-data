"""Offline review processing. Run: python -m src.pipeline --input data/raw/Reviews.csv"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ASPECTS = {
    "taste": ("taste", "flavor", "flavour", "delicious", "bland"),
    "price": ("price", "cost", "expensive", "cheap", "value"),
    "packaging": ("package", "packaging", "box", "sealed", "leak"),
    "delivery": ("delivery", "shipping", "arrived", "shipment", "late"),
    "freshness": ("fresh", "stale", "expired", "expiration"),
    "texture": ("texture", "crunchy", "chewy", "soft", "hard"),
}
POSITIVE = {"good", "great", "excellent", "love", "delicious", "fresh", "perfect", "fast", "best", "nice", "tasty"}
NEGATIVE = {"bad", "terrible", "awful", "hate", "stale", "expired", "broken", "slow", "worst", "poor", "bland", "disappointed"}
NEGATORS = {"not", "never", "no", "hardly"}
LABELS = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
COLUMNS = {"Id", "ProductId", "Score", "Time", "Text"}

def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def label_rating(rating: int) -> str:
    return LABELS[int(rating)]

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = frame.copy()
    df["review"] = df["Text"].map(clean_text)
    df["rating"] = pd.to_numeric(df["Score"], errors="coerce")
    df["date"] = pd.to_datetime(pd.to_numeric(df["Time"], errors="coerce"), unit="s", utc=True, errors="coerce")
    df = df[df["review"].str.len().ge(10) & df["rating"].isin(LABELS) & df["date"].notna()].copy()
    df["product_id"] = df["ProductId"].fillna("").astype(str)
    df["review_id"] = df["Id"].astype(str)
    df["label"] = df["rating"].astype(int).map(LABELS)
    # Repeated text must not appear on both sides of the time split.
    df["normalized"] = df["review"].str.lower().str.replace(r"\s+", " ", regex=True)
    df = df.sort_values(["date", "review_id"]).drop_duplicates(subset=["normalized"]).drop_duplicates(subset=["review_id"])
    return df[["review_id", "product_id", "date", "review", "rating", "label"]].reset_index(drop=True)

def aspect_sentiment(text: str) -> list[tuple[str, str]]:
    """Sentence-local lexical heuristic, NOT a trained aspect-based sentiment model."""
    output: list[tuple[str, str]] = []
    for sentence in re.split(r"[.!?;]+", text.lower()):
        tokens = re.findall(r"[a-z']+", sentence)
        aspects = [key for key, words in ASPECTS.items() if set(tokens) & set(words)]
        if not aspects:
            continue
        score = 0
        for i, token in enumerate(tokens):
            polarity = int(token in POSITIVE) - int(token in NEGATIVE)
            if polarity and any(n in NEGATORS for n in tokens[max(0, i - 3):i]):
                polarity *= -1
            score += polarity
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "unknown"
        output.extend((a, sentiment) for a in aspects)
    # Collapse repeated aspect mentions without inventing a majority sentiment.
    grouped: dict[str, set[str]] = {}
    for aspect, sentiment in output:
        grouped.setdefault(aspect, set()).add(sentiment)
    return [(a, next(iter(values)) if len(values) == 1 else "mixed") for a, values in grouped.items()]

def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["date", "review_id"]).reset_index(drop=True)
    if len(ordered) < 100:
        raise ValueError("At least 100 cleaned reviews are needed for evaluation")
    a, b = int(len(ordered) * .7), int(len(ordered) * .85)
    parts = (ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:])
    if any(p["label"].nunique() != 3 for p in parts):
        raise ValueError("Every chronological partition needs all three proxy-label classes")
    return parts

def build(input_path: Path, output: Path, limit: int = 12000, topics: int = 8) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if topics < 2 or limit < 100:
        raise ValueError("Use at least 2 topics and 100 samples")
    # Read source in chunks, then retain a reproducible chronological sample.
    batches = []
    for chunk in pd.read_csv(input_path, chunksize=40000, usecols=lambda c: c in COLUMNS, low_memory=False):
        batches.append(prepare(chunk))
    if not batches:
        raise ValueError("Dataset is empty")
    df = pd.concat(batches, ignore_index=True)
    df = df.drop_duplicates("review", keep="first").sort_values(["date", "review_id"])
    if len(df) > limit:
        # Evenly spaced observations retain approximate time coverage, not class balance.
        df = df.iloc[np.linspace(0, len(df) - 1, limit, dtype=int)]
    df = df.reset_index(drop=True)
    train, valid, test = split_chronological(df)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98, max_features=30000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(train["review"], train["label"])
    metrics = {"source_rows_after_cleaning_and_sampling": len(df), "train": len(train), "validation": len(valid), "test": len(test), "label_type": "rating-derived proxy, not human sentiment annotation"}
    for name, subset in (("validation", valid), ("test", test)):
        prediction = model.predict(subset["review"])
        metrics[name + "_macro_f1"] = float(f1_score(subset["label"], prediction, average="macro"))
        metrics[name + "_report"] = classification_report(subset["label"], prediction, labels=["negative", "neutral", "positive"], output_dict=True, zero_division=0)
        metrics[name + "_confusion"] = confusion_matrix(subset["label"], prediction, labels=["negative", "neutral", "positive"]).tolist()
    joblib.dump(model, output / "sentiment.joblib")
    # Topic model fitted on training only, avoiding test leakage into learned vocabulary.
    tfidf = TfidfVectorizer(stop_words="english", min_df=3, max_df=.95, max_features=6000)
    train_matrix = tfidf.fit_transform(train["review"])
    nmf = NMF(n_components=topics, init="nndsvda", random_state=42, max_iter=350)
    nmf.fit(train_matrix)
    all_topics = nmf.transform(tfidf.transform(df["review"]))
    names = tfidf.get_feature_names_out()
    topic_keywords = {str(i): [str(names[j]) for j in np.argsort(nmf.components_[i])[-8:][::-1]] for i in range(topics)}
    df["topic"] = all_topics.argmax(axis=1).astype(int)
    df["predicted_sentiment"] = model.predict(df["review"])
    df["split"] = np.select([df["review_id"].isin(train["review_id"]), df["review_id"].isin(valid["review_id"])], ["train", "validation"], default="test")
    aspect_rows = [(rid, a, s) for rid, review in df[["review_id", "review"]].itertuples(index=False, name=None) for a, s in aspect_sentiment(review)]
    aspects = pd.DataFrame(aspect_rows, columns=["review_id", "aspect", "aspect_sentiment"])
    con = duckdb.connect(str(output / "reviews.duckdb"))
    try:
        con.register("input_reviews", df.drop(columns=["rating"]).assign(rating=df["rating"].astype(int)))
        con.execute("CREATE OR REPLACE TABLE reviews AS SELECT * FROM input_reviews")
        con.register("input_aspects", aspects)
        con.execute("CREATE OR REPLACE TABLE aspects AS SELECT * FROM input_aspects")
    finally:
        con.close()
    metrics["aspect_coverage"] = float(aspects["review_id"].nunique() / len(df)) if len(aspects) else 0.
    metrics["topic_keywords"] = topic_keywords
    metrics["data_min_date"] = str(df["date"].min())
    metrics["data_max_date"] = str(df["date"].max())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} reviews and {len(aspects)} aspect records to {output}")
    print(f"Test macro F1 (rating proxy): {metrics['test_macro_f1']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/Reviews.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--topics", type=int, default=8)
    args = parser.parse_args()
    build(args.input, args.output, args.limit, args.topics)
