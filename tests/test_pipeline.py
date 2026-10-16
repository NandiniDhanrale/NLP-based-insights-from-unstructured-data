import pandas as pd
import pytest
from src.pipeline import aspect_sentiment, clean_text, label_rating, prepare, split_chronological

def test_clean_preserves_negation():
    assert clean_text(" <b>not</b> good   at all ") == "not good at all"

def test_rating_proxy():
    assert [label_rating(i) for i in (1, 2, 3, 4, 5)] == ["negative", "negative", "neutral", "positive", "positive"]

def test_prepare_rejects_blanks_and_duplicates():
    f = pd.DataFrame({"Id": [1, 2, 3], "ProductId": ["A"] * 3, "Score": [5, 1, 5], "Time": [1300000000] * 3, "Text": ["Not good at all", "Not good at all", ""]})
    result = prepare(f)
    assert len(result) == 1
    assert result.iloc[0]["label"] == "positive"

def test_missing_columns():
    with pytest.raises(ValueError, match="Missing"):
        prepare(pd.DataFrame({"Text": ["example"]}))

def test_aspect_local_sentiment_and_unknown():
    assert ("delivery", "negative") in aspect_sentiment("Delivery was not good.")
    assert ("price", "unknown") in aspect_sentiment("Price was mentioned.")
    assert aspect_sentiment("No aspect terms here.") == []

def test_split_has_no_review_overlap():
    f = pd.DataFrame({"review_id": [str(i) for i in range(120)], "date": pd.date_range("2020-01-01", periods=120, tz="UTC"), "label": ["negative", "neutral", "positive"] * 40})
    a, b, c = split_chronological(f)
    assert len(a) + len(b) + len(c) == 120
    assert set(a.review_id).isdisjoint(b.review_id)
    assert set(b.review_id).isdisjoint(c.review_id)
