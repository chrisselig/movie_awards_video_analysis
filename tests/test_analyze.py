"""Tests for analyze_data.py analysis functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analyze_data import (
    tokenize, analyze_word_frequency, analyze_filler_words,
    analyze_superlatives, analyze_agreement_disagreement,
    analyze_hot_takes, analyze_vocabulary_richness,
    get_segments_for_speaker
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
