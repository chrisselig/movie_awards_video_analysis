"""Tests for analyze_data.py analysis functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analyze_data import (
    tokenize, analyze_word_frequency, analyze_filler_words,
    analyze_superlatives, analyze_agreement_disagreement,
    analyze_hot_takes, analyze_vocabulary_richness,
    analyze_castle, get_segments_for_speaker,
    analyze_sentiment, analyze_rants, build_markov_chain,
    compute_tfidf_fingerprints, train_castle_predictor,
    compute_greatest_rants,
)


def _make_segments(text, speaker=None):
    return [{"start": 0, "duration": 10, "text": text, "speaker": speaker}]


def test_tokenize_extracts_lowercase_words():
    assert tokenize("Hello World 123") == ["hello", "world"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_analyze_word_frequency_removes_stop_words():
    segs = _make_segments("the the the movie movie movie film film great")
    result = analyze_word_frequency(segs)
    words = [w for w, _ in result]
    assert "the" not in words
    assert "movie" in words


def test_analyze_filler_words_counts_correctly():
    segs = _make_segments("like um like literally like honestly")
    result = analyze_filler_words(segs, 60)
    assert result["like"]["count"] == 3
    assert result["um"]["count"] == 1
    assert result["literally"]["count"] == 1


def test_analyze_filler_words_empty_transcript():
    segs = _make_segments("")
    result = analyze_filler_words(segs, 60)
    assert len(result) == 0


def test_analyze_superlatives_counts_correctly():
    segs = _make_segments("best best worst amazing terrible masterpiece")
    result = analyze_superlatives(segs, 120)
    assert result["total"] == 6
    assert result["words"]["best"]["count"] == 2
    assert result["words"]["worst"]["count"] == 1


def test_analyze_superlatives_calculates_per_minute():
    segs = _make_segments("best worst amazing")
    result = analyze_superlatives(segs, 60)
    assert result["per_minute"] == 3.0


def test_analyze_agreement_disagreement_counts():
    segs = _make_segments("i agree exactly i disagree no way")
    result = analyze_agreement_disagreement(segs)
    assert result["agreements"] == 2
    assert result["disagreements"] == 2


def test_analyze_agreement_disagreement_empty():
    segs = _make_segments("hello world nothing here")
    result = analyze_agreement_disagreement(segs)
    assert result["agreements"] == 0
    assert result["disagreements"] == 0


def test_analyze_hot_takes_detects_markers():
    segs = _make_segments("this movie is overrated and honestly underrated at the same time")
    result = analyze_hot_takes(segs)
    assert result["count"] == 2


def test_analyze_hot_takes_no_markers():
    segs = _make_segments("this movie was really nice")
    result = analyze_hot_takes(segs)
    assert result["count"] == 0


def test_analyze_vocabulary_richness():
    segs = _make_segments("the cat sat on the mat the cat")
    result = analyze_vocabulary_richness(segs, 60)
    assert result["total_words"] == 8
    assert result["unique_words"] == 5
    assert result["ttr"] == 0.625


def test_analyze_vocabulary_richness_empty():
    segs = _make_segments("")
    result = analyze_vocabulary_richness(segs, 60)
    assert result["total_words"] == 0
    assert result["ttr"] == 0


def test_analyze_castle_counts_mentions():
    segs = _make_segments("the Castle was great and Castle Award goes to this film")
    result = analyze_castle(segs)
    assert result["total_mentions"] == 2
    assert result["castle_award_mentions"] == 1


def test_analyze_castle_excludes_infinity_castle():
    segs = _make_segments("Infinity Castle was a great movie with castle vibes")
    result = analyze_castle(segs)
    assert result["movie_castle_mentions"] == 1
    assert result["rabbit_mentions"] == 1


def test_analyze_castle_no_mentions():
    segs = _make_segments("this movie was just okay nothing special")
    result = analyze_castle(segs)
    assert result["total_mentions"] == 0


def test_get_segments_for_speaker_filters():
    segs = [
        {"start": 0, "duration": 5, "text": "hello", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "world", "speaker": "Tyler"},
        {"start": 10, "duration": 5, "text": "foo", "speaker": "Justin"},
    ]
    video = {"segments": segs}
    justin = get_segments_for_speaker(video, "Justin")
    assert len(justin) == 2
    tyler = get_segments_for_speaker(video, "Tyler")
    assert len(tyler) == 1


def test_get_segments_for_speaker_none_returns_all():
    segs = [
        {"start": 0, "duration": 5, "text": "a", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "b", "speaker": "Tyler"},
    ]
    video = {"segments": segs}
    assert len(get_segments_for_speaker(video, None)) == 2


# --- Sentiment Analysis Tests ---

def test_sentiment_returns_compound_score():
    segs = _make_segments("This movie was absolutely amazing and wonderful")
    result = analyze_sentiment(segs, 60)
    assert result["compound"] > 0
    assert "pos" in result
    assert "neg" in result
    assert "neu" in result


def test_sentiment_negative_text():
    segs = _make_segments("This was terrible awful garbage horrible trash")
    result = analyze_sentiment(segs, 60)
    assert result["compound"] < 0


def test_sentiment_timeline_has_10_chunks():
    segs = [{"start": i * 6, "duration": 5, "text": "movie is good", "speaker": None} for i in range(10)]
    result = analyze_sentiment(segs, 60)
    assert len(result["timeline"]) == 10


def test_sentiment_most_positive_and_negative():
    segs = [
        {"start": 0, "duration": 5, "text": "This is absolutely wonderful and amazing", "speaker": None},
        {"start": 10, "duration": 5, "text": "This is terrible horrible garbage", "speaker": None},
    ]
    result = analyze_sentiment(segs, 20)
    assert result["most_positive"] is not None
    assert result["most_negative"] is not None
    assert result["most_positive"]["compound"] > result["most_negative"]["compound"]


def test_sentiment_empty_segments():
    result = analyze_sentiment([], 60)
    assert result["compound"] == 0
    assert result["timeline"] == []


def test_sentiment_preserves_original_text():
    segs = [{"start": 0, "duration": 5, "text": "This Movie Was GREAT!", "speaker": None}]
    result = analyze_sentiment(segs, 10)
    assert result["most_positive"]["text"] == "This Movie Was GREAT!"


# --- Rant Detector Tests ---

def test_rant_detector_returns_rant_index():
    segs = [{"start": i, "duration": 1, "text": "terrible awful garbage worst insane " * 5, "speaker": None} for i in range(30)]
    result = analyze_rants(segs, 30, {})
    assert "rant_index" in result
    assert "top_rants" in result


def test_rant_detector_empty_segments():
    result = analyze_rants([], 60, {})
    assert result["rant_index"] == 0
    assert result["top_rants"] == []


def test_rant_detector_top_rants_have_required_fields():
    segs = [{"start": i, "duration": 1, "text": "um like literally terrible worst garbage insane amazing " * 3, "speaker": None} for i in range(60)]
    result = analyze_rants(segs, 60, {})
    if result["top_rants"]:
        rant = result["top_rants"][0]
        assert "start" in rant
        assert "score" in rant
        assert "wpm" in rant
        assert "text_preview" in rant


def test_rant_preserves_original_text():
    segs = [{"start": i, "duration": 1, "text": "TERRIBLE Awful GARBAGE! " * 5, "speaker": None} for i in range(30)]
    result = analyze_rants(segs, 30, {})
    if result["top_rants"]:
        assert "TERRIBLE" in result["top_rants"][0]["text_preview"]


def test_compute_greatest_rants_ranks_across_videos():
    video_analyses = {
        "v1": {"title": "Vid A", "year": 2020, "rants": {"rant_index": 1.5, "top_rants": [
            {"start": 10, "score": 1.5, "wpm": 200, "neg": 0.3, "sup_density": 5, "filler_density": 3, "text_preview": "rant 1"}
        ]}},
        "v2": {"title": "Vid B", "year": 2021, "rants": {"rant_index": 2.0, "top_rants": [
            {"start": 20, "score": 2.0, "wpm": 250, "neg": 0.4, "sup_density": 6, "filler_density": 4, "text_preview": "rant 2"}
        ]}},
    }
    result = compute_greatest_rants(video_analyses)
    assert len(result) == 2
    assert result[0]["score"] > result[1]["score"]
    assert result[0]["title"] == "Vid B"


# --- Markov Chain Tests ---

def test_markov_chain_generates_quotes():
    segs = [{"start": i, "duration": 1, "text": f"the movie was really great and the acting was superb word{i}"} for i in range(50)]
    quotes = build_markov_chain(segs, order=2)
    assert isinstance(quotes, list)


def test_markov_chain_empty_input():
    quotes = build_markov_chain([], order=2)
    assert quotes == []


def test_markov_chain_reproducible():
    segs = [{"start": i, "duration": 1, "text": f"every single movie this year was incredible and honestly amazing word{i}"} for i in range(100)]
    q1 = build_markov_chain(segs, order=2)
    q2 = build_markov_chain(segs, order=2)
    assert q1 == q2


# --- TF-IDF Tests ---

def test_tfidf_returns_year_signatures():
    all_data = {
        "v1": {"year": 2020, "segments": [{"text": "batman superman spider action hero"}]},
        "v2": {"year": 2021, "segments": [{"text": "romance comedy drama love story"}]},
    }
    result = compute_tfidf_fingerprints(all_data)
    assert "year_signatures" in result
    assert "word_trends" in result
    assert 2020 in result["year_signatures"] or "2020" in result["year_signatures"]


def test_tfidf_empty_data():
    result = compute_tfidf_fingerprints({})
    assert result["year_signatures"] == {}


def test_tfidf_tracks_words():
    all_data = {
        "v1": {"year": 2020, "segments": [{"text": "literally amazing incredible masterpiece " * 10}]},
        "v2": {"year": 2021, "segments": [{"text": "terrible insane honestly honestly honestly " * 10}]},
    }
    result = compute_tfidf_fingerprints(all_data)
    # Should have tracked at least some of the specified words
    assert isinstance(result["word_trends"], dict)


# --- Castle Predictor Tests ---

def test_castle_predictor_not_enough_data():
    # Less than 10 videos = error
    analyses = {f"v{i}": {
        "castle": {"rabbit_mentions": 0},
        "vocabulary": {"total_words": 100, "words_per_minute": 150},
        "superlatives": {"hyperbole_index": 5},
        "filler_words": {"um": {"count": 3}},
        "hot_takes": {"count": 1},
        "year": 2020,
        "duration_min": 30,
    } for i in range(5)}
    result = train_castle_predictor(analyses)
    assert "error" in result


def test_castle_predictor_with_enough_data():
    analyses = {}
    for i in range(20):
        analyses[f"v{i}"] = {
            "castle": {"rabbit_mentions": 1 if i % 3 == 0 else 0},
            "vocabulary": {"total_words": 500 + i * 10, "words_per_minute": 140 + i},
            "superlatives": {"hyperbole_index": 3 + i * 0.5},
            "filler_words": {"um": {"count": 5 + i}, "like": {"count": 10 + i}},
            "hot_takes": {"count": i % 4},
            "year": 2015 + i % 10,
            "duration_min": 25 + i,
        }
    result = train_castle_predictor(analyses)
    assert "coefficients" in result
    assert len(result["coefficients"]) == 7
    assert 0 <= result["accuracy"] <= 1
    assert len(result["probabilities"]) == 20
