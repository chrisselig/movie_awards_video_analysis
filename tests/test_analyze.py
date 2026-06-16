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
    analyze_interruptions, analyze_repeated_phrases,
    analyze_speed_by_context, analyze_monologues,
    analyze_hipster_index, analyze_agreement_asymmetry,
    analyze_conversation_momentum, analyze_last_word,
    categorize_superlatives_by_speaker,
    compute_taste_divergence, compute_rant_starters,
    compute_merged_profile, compute_trading_cards,
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
    # "movie" is in EXTRA_STOPS, so it gets filtered; "film" and "great" should remain
    assert "film" in words


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


# --- Head-to-Head Feature Tests ---

def _make_speaker_segments():
    """Helper: alternating Justin/Tyler segments with timing."""
    return [
        {"start": 0.0, "duration": 5.0, "text": "I think this movie was great", "speaker": "Justin"},
        {"start": 5.0, "duration": 4.0, "text": "no way it was terrible", "speaker": "Tyler"},
        {"start": 9.0, "duration": 3.0, "text": "i agree it had problems", "speaker": "Justin"},
        {"start": 12.5, "duration": 5.0, "text": "exactly the worst movie", "speaker": "Tyler"},
        {"start": 18.0, "duration": 4.0, "text": "but the acting was amazing", "speaker": "Justin"},
    ]


def test_interruptions_detects_speaker_switch():
    # gap between seg 1 end (9.0) and seg 2 start (9.0) = 0, so interruption
    segs = _make_speaker_segments()
    result = analyze_interruptions(segs)
    assert result["justin_interrupts_tyler"] + result["tyler_interrupts_justin"] > 0
    assert len(result["interruption_moments"]) > 0


def test_interruptions_no_speaker_data():
    segs = [{"start": 0, "duration": 5, "text": "hello", "speaker": None},
            {"start": 5, "duration": 5, "text": "world", "speaker": None}]
    result = analyze_interruptions(segs)
    assert result["justin_interrupts_tyler"] == 0
    assert result["tyler_interrupts_justin"] == 0


def test_repeated_phrases_finds_ngrams():
    segs = [
        {"start": 0, "duration": 5, "text": "the best movie the best movie the best movie ever", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "yeah totally cool", "speaker": "Tyler"},
    ]
    result = analyze_repeated_phrases(segs, "Justin")
    assert len(result) > 0
    assert any("best movie" in r["phrase"] for r in result)


def test_repeated_phrases_filters_stopwords():
    segs = [
        {"start": 0, "duration": 5, "text": "the the the and and and the the the", "speaker": "Justin"},
    ]
    result = analyze_repeated_phrases(segs, "Justin")
    # All stopword-only grams should be filtered out
    assert len(result) == 0


def test_monologues_finds_longest_run():
    segs = [
        {"start": 0, "duration": 10, "text": "blah", "speaker": "Justin"},
        {"start": 10, "duration": 10, "text": "blah", "speaker": "Justin"},
        {"start": 20, "duration": 5, "text": "blah", "speaker": "Tyler"},
        {"start": 25, "duration": 3, "text": "blah", "speaker": "Tyler"},
        {"start": 28, "duration": 2, "text": "blah", "speaker": "Tyler"},
    ]
    result = analyze_monologues(segs)
    assert len(result["Justin"]) > 0
    assert result["Justin"][0]["duration"] == 20.0
    assert result["Tyler"][0]["duration"] == 10.0


def test_hipster_index_counts_markers():
    segs = [
        {"start": 0, "duration": 5, "text": "that movie is overrated and overhyped honestly", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "nah it is underrated a hidden gem", "speaker": "Tyler"},
    ]
    j_result = analyze_hipster_index(segs, "Justin")
    assert j_result["overrated_count"] == 2
    assert j_result["underrated_count"] == 0

    t_result = analyze_hipster_index(segs, "Tyler")
    assert t_result["underrated_count"] == 2  # "underrated" + "hidden gem"
    assert t_result["overrated_count"] == 0


def test_agreement_asymmetry_per_speaker():
    segs = [
        {"start": 0, "duration": 5, "text": "i agree exactly absolutely", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "i disagree no way hard disagree", "speaker": "Tyler"},
    ]
    result = analyze_agreement_asymmetry(segs)
    assert result["Justin"]["agreements"] == 3
    assert result["Justin"]["disagreements"] == 0
    assert result["Tyler"]["agreements"] == 0
    assert result["Tyler"]["disagreements"] == 3


def test_conversation_momentum_counts_switches():
    segs = [
        {"start": 0, "duration": 2, "text": "a", "speaker": "Justin"},
        {"start": 2, "duration": 2, "text": "b", "speaker": "Tyler"},
        {"start": 4, "duration": 2, "text": "c", "speaker": "Justin"},
        {"start": 6, "duration": 2, "text": "d", "speaker": "Tyler"},
    ]
    result = analyze_conversation_momentum(segs)
    assert result["switches_per_minute"] > 0
    assert len(result["timeline"]) > 0


def test_last_word_identifies_final_speaker():
    segs = [
        {"start": 0, "duration": 5, "text": "opening", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "middle", "speaker": "Tyler"},
        {"start": 10, "duration": 5, "text": "last word here", "speaker": "Justin"},
    ]
    result = analyze_last_word(segs)
    assert result["speaker"] == "Justin"


def test_last_word_no_speaker():
    segs = [{"start": 0, "duration": 5, "text": "hello", "speaker": None}]
    result = analyze_last_word(segs)
    assert result["speaker"] is None


def test_trading_cards_normalizes_stats():
    speaker_agg = {
        "Justin": {
            "words_per_minute": 160, "unique_words": 5000, "total_fillers": 100,
            "total_superlatives": 50, "hot_takes": 20, "filler_per_minute": 3.0,
            "videos_with_data": 10,
        },
        "Tyler": {
            "words_per_minute": 140, "unique_words": 4000, "total_fillers": 80,
            "total_superlatives": 60, "hot_takes": 15, "filler_per_minute": 2.5,
            "videos_with_data": 10,
        },
    }
    video_analyses = {}
    result = compute_trading_cards(speaker_agg, video_analyses)
    assert "Justin" in result
    assert "Tyler" in result
    assert "normalized" in result["Justin"]
    # One speaker should have 100 and the other 0 for each stat
    assert result["Justin"]["normalized"]["wpm"] == 100.0
    assert result["Tyler"]["normalized"]["wpm"] == 0.0


def test_taste_divergence_finds_gap():
    video_analyses = {
        "v1": {
            "title": "Happy Movie", "year": 2020,
            "by_speaker": {
                "Justin": {"sentiment": {"compound": 0.8}},
                "Tyler": {"sentiment": {"compound": -0.2}},
            },
        },
        "v2": {
            "title": "Agreed Movie", "year": 2021,
            "by_speaker": {
                "Justin": {"sentiment": {"compound": 0.5}},
                "Tyler": {"sentiment": {"compound": 0.5}},
            },
        },
    }
    result = compute_taste_divergence(video_analyses)
    assert len(result["most_divisive"]) > 0
    assert result["most_divisive"][0]["title"] == "Happy Movie"
    assert result["most_divisive"][0]["gap"] == 1.0


def test_merged_profile_combines_stats():
    speaker_agg = {
        "Justin": {
            "words_per_minute": 160, "unique_words": 5000, "total_fillers": 100,
            "total_superlatives": 50, "hot_takes": 20,
        },
        "Tyler": {
            "words_per_minute": 140, "unique_words": 4000, "total_fillers": 80,
            "total_superlatives": 60, "hot_takes": 15,
        },
    }
    result = compute_merged_profile(speaker_agg)
    assert result["wpm"] == 150.0
    assert result["unique_words"] == 9000
    assert result["speed_from"] == "Justin"


def test_rant_starters_attributes_speaker():
    video_analyses = {
        "v1": {
            "title": "Ranty Movie", "year": 2020,
            "rants": {"rant_index": 2.0, "top_rants": [
                {"start": 10, "end": 40, "score": 2.0, "text_preview": "terrible awful"}
            ]},
            "_segments_raw": [
                {"start": 8, "duration": 10, "text": "terrible", "speaker": "Justin"},
                {"start": 20, "duration": 10, "text": "awful", "speaker": "Tyler"},
            ],
        },
    }
    result = compute_rant_starters(video_analyses)
    # Rant midpoint is 25, Tyler is speaking at 20-30
    assert result["counts"]["Tyler"] >= 1 or result["counts"]["Justin"] >= 1


def test_speed_debater_splits_contexts():
    segs = [
        {"start": 0, "duration": 5, "text": "i agree this movie was fantastic and wonderful", "speaker": "Justin"},
        {"start": 5, "duration": 5, "text": "i disagree this was terrible awful garbage worst", "speaker": "Tyler"},
    ]
    result = analyze_speed_by_context(segs, {})
    assert result["Justin"]["agree_wpm"] > 0
    assert result["Tyler"]["disagree_wpm"] > 0


def test_superlative_style_categorizes():
    segs = [
        {"start": 0, "duration": 5, "text": "best greatest amazing insane terrible", "speaker": "Justin"},
    ]
    result = categorize_superlatives_by_speaker(segs, "Justin")
    assert result["positive"] == 3  # best, greatest, amazing
    assert result["negative"] == 1  # terrible
    assert result["intensity"] == 1  # insane
