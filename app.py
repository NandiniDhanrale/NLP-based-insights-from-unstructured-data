"""Streamlit dashboard for offline-computed Customer Voice Intelligence artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
st.set_page_config(page_title="Customer Voice Intelligence", page_icon="📊", layout="wide")
st.title("Customer Voice Intelligence")
st.caption("Review-derived insights | Amazon Fine Food Reviews, historical data through 2012 | ratings are proxy labels")

@st.cache_data(show_spinner=False)
def load_tables():
    con = duckdb.connect(str(ARTIFACTS / "reviews.duckdb"), read_only=True)
    try:
        reviews = con.execute("SELECT * FROM reviews").df()
        aspects = con.execute("SELECT * FROM aspects").df()
    finally:
        con.close()
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    reviews["date"] = pd.to_datetime(reviews["date"], utc=True)
    return reviews, aspects, metrics

required = [ARTIFACTS / "reviews.duckdb", ARTIFACTS / "metrics.json", ARTIFACTS / "sentiment.joblib"]
if not all(p.exists() for p in required):
    st.warning("Analysis artifacts are not available. This is a project source checkout, not a completed trained dashboard.")
    st.markdown("Download the CC0 Amazon Fine Food Reviews `Reviews.csv` from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews), place it at `data/raw/Reviews.csv`, and run `python -m src.pipeline`. The artifacts directory must exist in the deployed repository or a persistent artifact store.")
    st.stop()

reviews, aspects, metrics = load_tables()
page = st.sidebar.radio("Page", ["Executive Overview", "Sentiment Analytics", "Topic Discovery", "Aspect Insights", "Model Evaluation", "Review Explorer"])
start, end = reviews["date"].min().date(), reviews["date"].max().date()
period = st.sidebar.date_input("Review dates", value=(start, end), min_value=start, max_value=end)
if isinstance(period, tuple) and len(period) == 2:
    reviews = reviews[reviews["date"].dt.date.between(period[0], period[1])]
products = sorted(reviews["product_id"].dropna().unique().tolist())
selected_products = st.sidebar.multiselect("Product IDs", products, max_selections=20)
if selected_products:
    reviews = reviews[reviews["product_id"].isin(selected_products)]
labels = st.sidebar.multiselect("Predicted sentiment", ["negative", "neutral", "positive"])
if labels:
    reviews = reviews[reviews["predicted_sentiment"].isin(labels)]
topics = sorted(reviews["topic"].dropna().unique().tolist())
topic = st.sidebar.selectbox("Topic", ["All"] + [str(t) for t in topics])
if topic != "All":
    reviews = reviews[reviews["topic"] == int(topic)]
selected_aspects = st.sidebar.multiselect("Aspect", sorted(aspects["aspect"].unique().tolist()))
if selected_aspects:
    ids = aspects.loc[aspects["aspect"].isin(selected_aspects), "review_id"]
    reviews = reviews[reviews["review_id"].isin(ids)]
aspects = aspects[aspects["review_id"].isin(reviews["review_id"])]
st.caption(f"Filtered sample: {len(reviews):,} reviews. All metrics on data pages use this filter, except the fixed evaluation results.")
if reviews.empty:
    st.info("No reviews match the current filters.")
    st.stop()

def bar_counts(df, column, title):
    counts = df.groupby(column).size().rename("reviews").reset_index()
    st.plotly_chart(px.bar(counts, x=column, y="reviews", title=title), use_container_width=True)

def trend(df, group, title):
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M").astype(str)
    counts = x.groupby(["month", group]).size().rename("reviews").reset_index()
    st.plotly_chart(px.line(counts, x="month", y="reviews", color=group, title=title, markers=True), use_container_width=True)

if page == "Executive Overview":
    a, b, c, d = st.columns(4)
    a.metric("Reviews in filter", f"{len(reviews):,}")
    b.metric("Distinct products", f"{reviews['product_id'].nunique():,}")
    c.metric("Predicted negative", f"{(reviews['predicted_sentiment'] == 'negative').mean():.1%}")
    d.metric("Aspect coverage", f"{aspects['review_id'].nunique() / len(reviews):.1%}")
    left, right = st.columns(2)
    with left:
        bar_counts(reviews, "predicted_sentiment", "Model-predicted sentiment")
    with right:
        bar_counts(reviews, "topic", "NMF topic frequency")
    trend(reviews, "predicted_sentiment", "Predicted sentiment by month")
elif page == "Sentiment Analytics":
    st.subheader("Sentiment analysis")
    st.info("The classifier was trained on rating-derived proxy labels. This is not independent human-annotated sentiment ground truth. Predicted probabilities are not calibrated.")
    bar_counts(reviews, "predicted_sentiment", "Predicted class counts")
    trend(reviews, "predicted_sentiment", "Monthly distribution")
    st.dataframe(reviews[["date", "product_id", "rating", "label", "predicted_sentiment", "review"]].head(250), use_container_width=True)
elif page == "Topic Discovery":
    st.subheader("NMF topic discovery")
    keywords = metrics.get("topic_keywords", {})
    st.dataframe(pd.DataFrame([{"topic": int(k), "keywords (not validated topic names)": ", ".join(v)} for k, v in keywords.items()]), hide_index=True, use_container_width=True)
    bar_counts(reviews, "topic", "Topic frequency")
    trend(reviews, "topic", "Topics by month")
    for t in sorted(reviews["topic"].unique()):
        with st.expander(f"Topic {t}: representative reviews"):
            st.dataframe(reviews.loc[reviews["topic"] == t, ["review", "date"]].head(8), hide_index=True, use_container_width=True)
elif page == "Aspect Insights":
    st.subheader("Aspect-level lexical analysis")
    st.info("Sentence-local keyword and polarity rules. Unknown or mixed outcomes are retained. This is not an evaluated neural aspect-based sentiment model.")
    st.metric("Reviews with ≥1 identified aspect", f"{aspects['review_id'].nunique():,}")
    if len(aspects):
        summary = aspects.groupby(["aspect", "aspect_sentiment"]).size().rename("mentions").reset_index()
        st.plotly_chart(px.bar(summary, x="aspect", y="mentions", color="aspect_sentiment", barmode="stack", title="Aspect sentiment (heuristic)"), use_container_width=True)
        negative = aspects[aspects["aspect_sentiment"] == "negative"]
        st.subheader("Negative aspect mentions")
        st.dataframe(negative.merge(reviews[["review_id", "review"]], on="review_id").head(150), hide_index=True, use_container_width=True)
    else:
        st.info("No aspect mentions under these filters.")
elif page == "Model Evaluation":
    st.subheader("Chronological holdout evaluation")
    st.caption(f"Train: {metrics['train']:,} | Validation: {metrics['validation']:,} | Test: {metrics['test']:,}; uncalibrated TF-IDF + Logistic Regression; labels come from review stars.")
    left, right = st.columns(2)
    left.metric("Validation macro F1", f"{metrics['validation_macro_f1']:.3f}")
    right.metric("Held-out test macro F1", f"{metrics['test_macro_f1']:.3f}")
    st.dataframe(pd.DataFrame(metrics["test_report"]).T, use_container_width=True)
    matrix = pd.DataFrame(metrics["test_confusion"], index=["negative", "neutral", "positive"], columns=["negative", "neutral", "positive"])
    st.plotly_chart(px.imshow(matrix, text_auto=True, labels={"x": "Predicted", "y": "Rating proxy", "color": "Reviews"}, title="Held-out confusion matrix"), use_container_width=True)
    st.warning("Ratings can conflict with review text. Topic and aspect extraction have no gold-standard evaluation in this project. Transformer performance is not claimed.")
elif page == "Review Explorer":
    st.subheader("Review explorer")
    search = st.text_input("Find review text (literal substring)")
    found = reviews[reviews["review"].str.contains(search, case=False, regex=False, na=False)] if search else reviews
    st.dataframe(found[["review_id", "date", "product_id", "rating", "predicted_sentiment", "topic", "review"]].head(1000), hide_index=True, use_container_width=True)
    st.caption("Maximum 1,000 reviews shown. Filters apply before the search.")
    st.download_button("Download filtered reviews", data=found.to_csv(index=False), file_name="filtered_reviews.csv", mime="text/csv")

st.sidebar.divider()
with st.sidebar.expander("Predict a new review"):
    value = st.text_area("Review text", max_chars=2000)
    if st.button("Predict"):
        if len(value.strip()) < 10:
            st.error("Enter at least 10 characters.")
        else:
            @st.cache_resource(show_spinner=False)
            def load_model():
                return joblib.load(ARTIFACTS / "sentiment.joblib")
            st.success(f"Model prediction (rating-proxy classifier): {load_model().predict([value])[0]}")
            st.caption("This prediction does not represent calibrated confidence.")
