#!/usr/bin/env python3
"""
Analyze movie awards transcripts and generate JSON data for the D3.js infographic.

Supports per-speaker analysis when diarization data is available.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import libsql_experimental as libsql
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

load_dotenv()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("vader_lexicon", quiet=True)

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
STOP_WORDS = set(stopwords.words("english"))

EXTRA_STOPS = {"like", "know", "yeah", "okay", "oh", "well", "go", "going",
               "got", "get", "thing", "things", "really", "right", "uh",
               "um", "na", "gon", "wan", "one", "would", "could", "also",
               "think", "much", "good", "even", "way", "lot", "say", "said",
               "see", "still", "actually", "pretty", "make", "take", "come",
               "back", "something", "kind", "mean", "want", "let", "put",
               "look", "two", "first", "next", "number", "movie", "movies"}

FILLER_WORDS = ["um", "uh", "like", "literally", "honestly", "basically",
                "obviously", "actually", "seriously", "absolutely",
                "definitely", "totally", "essentially", "incredible",
                "insane", "amazing"]

SUPERLATIVES = ["best", "worst", "greatest", "amazing", "terrible", "incredible",
                "fantastic", "awful", "perfect", "masterpiece", "brilliant",
                "genius", "horrible", "beautiful", "stunning", "phenomenal",
                "outstanding", "atrocious", "garbage", "trash", "flawless",
                "insane", "mind-blowing", "goat"]

AGREEMENT_PHRASES = ["i agree", "exactly", "absolutely", "100%", "hundred percent",
                     "for sure", "same", "oh yeah", "true", "that's true",
                     "good point", "you're right"]
DISAGREEMENT_PHRASES = ["i disagree", "no way", "nah", "i don't think",
                        "that's wrong", "come on", "what are you", "are you kidding",
                        "you're crazy", "hard disagree", "strongly disagree",
                        "i don't know about that"]

HOT_TAKE_MARKERS = ["hot take", "unpopular opinion", "controversial",
                    "fight me", "i don't care what anyone", "overrated",
                    "underrated", "overhyped", "not that good", "mid",
                    "i said what i said", "sue me"]

REACTION_PATTERNS = {
    "laughter": re.compile(r'\[(?:laughter|laughing|laughs|ha ha)\]', re.I),
    "music": re.compile(r'\[(?:music|applause)\]', re.I),
    "crosstalk": re.compile(r'\[(?:crosstalk|inaudible|overlapping)\]', re.I),
}

SPEAKERS = ["Justin", "Tyler"]

# Castle the Rabbit - Tyler's rabbit named after Frank Castle (The Punisher)
CASTLE_PATTERNS = re.compile(r'\bcastle\b', re.I)


def log(msg):
    print(msg, flush=True)


def get_db():
    return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)


def load_all_data(conn):
    """Load all videos and their transcripts with optional speaker labels."""
    videos = conn.execute(
        "SELECT video_id, title, duration_seconds, year, category, upload_date FROM videos ORDER BY year, title"
    ).fetchall()

    # Check if speaker column exists
    has_speaker = False
    try:
        conn.execute("SELECT speaker FROM transcript_segments LIMIT 1")
        has_speaker = True
    except Exception:
        pass

    # Check if speaker_segments table exists
    has_diarization = False
    try:
        conn.execute("SELECT COUNT(*) FROM speaker_segments")
        has_diarization = True
    except Exception:
        pass

    data = {}
    for vid_id, title, duration, year, category, upload_date in videos:
        if has_speaker:
            segments = conn.execute(
                "SELECT start_time, duration, text, speaker FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
                (vid_id,)
            ).fetchall()
            seg_list = [{"start": s[0], "duration": s[1], "text": s[2], "speaker": s[3]} for s in segments]
        else:
            segments = conn.execute(
                "SELECT start_time, duration, text FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
                (vid_id,)
            ).fetchall()
            seg_list = [{"start": s[0], "duration": s[1], "text": s[2], "speaker": None} for s in segments]

        # Load speaker timing data if available
        speaker_times = {}
        if has_diarization:
            sp_rows = conn.execute(
                "SELECT speaker_label, SUM(duration) FROM speaker_segments WHERE video_id = ? GROUP BY speaker_label",
                (vid_id,)
            ).fetchall()
            speaker_times = {r[0]: round(r[1], 1) for r in sp_rows}

        data[vid_id] = {
            "video_id": vid_id,
            "title": title,
            "duration": duration or 0,
            "year": year,
            "category": category,
            "upload_date": upload_date,
            "segments": seg_list,
            "speaker_times": speaker_times,
        }
    return data


def get_segments_for_speaker(video_data, speaker=None):
    """Filter segments by speaker. None = all segments."""
    if speaker is None:
        return video_data["segments"]
    return [s for s in video_data["segments"] if s.get("speaker") == speaker]


def text_from_segments(segments):
    return " ".join(s["text"] for s in segments).lower()


def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())


def analyze_word_frequency(segments):
    text = text_from_segments(segments)
    words = tokenize(text)
    filtered = [w for w in words if w not in STOP_WORDS and w not in EXTRA_STOPS and len(w) > 2]
    return Counter(filtered).most_common(30)


def analyze_filler_words(segments, duration_seconds):
    text = text_from_segments(segments)
    words = tokenize(text)
    total_words = len(words)
    duration_min = max(duration_seconds / 60, 1)

    counts = {}
    for filler in FILLER_WORDS:
        count = words.count(filler)
        if count > 0:
            counts[filler] = {
                "count": count,
                "per_minute": round(count / duration_min, 2),
                "pct_of_words": round(count / max(total_words, 1) * 100, 3)
            }
    return counts


def analyze_superlatives(segments, duration_seconds):
    text = text_from_segments(segments)
    words = tokenize(text)
    duration_min = max(duration_seconds / 60, 1)

    counts = {}
    for sup in SUPERLATIVES:
        count = words.count(sup)
        if count > 0:
            counts[sup] = {"count": count, "per_minute": round(count / duration_min, 2)}

    total = sum(v["count"] for v in counts.values())
    return {
        "words": counts,
        "total": total,
        "per_minute": round(total / duration_min, 2),
        "hyperbole_index": round(total / max(len(words), 1) * 1000, 2)
    }


def analyze_agreement_disagreement(segments):
    text = text_from_segments(segments)
    agrees = sum(text.count(phrase) for phrase in AGREEMENT_PHRASES)
    disagrees = sum(text.count(phrase) for phrase in DISAGREEMENT_PHRASES)
    total = agrees + disagrees
    return {
        "agreements": agrees,
        "disagreements": disagrees,
        "harmony_ratio": round(agrees / max(disagrees, 1), 2),
        "total_opinions": total
    }


def analyze_hot_takes(segments):
    text = text_from_segments(segments)
    hot_takes = []
    for marker in HOT_TAKE_MARKERS:
        for match in re.finditer(re.escape(marker), text):
            pos = match.start()
            context_start = max(0, pos - 50)
            context_end = min(len(text), pos + 100)
            hot_takes.append({
                "marker": marker,
                "context": text[context_start:context_end].strip()
            })
    return {"count": len(hot_takes), "instances": hot_takes[:10]}


def analyze_vocabulary_richness(segments, duration_seconds):
    text = text_from_segments(segments)
    words = tokenize(text)
    if not words:
        return {"ttr": 0, "unique_words": 0, "total_words": 0, "words_per_minute": 0}
    unique = set(words)
    return {
        "ttr": round(len(unique) / len(words), 4),
        "unique_words": len(unique),
        "total_words": len(words),
        "words_per_minute": round(len(words) / max(duration_seconds / 60, 1), 1)
    }


def analyze_energy_over_time(video_data):
    segments = video_data["segments"]
    if not segments:
        return []
    duration = video_data["duration"]
    if duration <= 0:
        return []

    n_chunks = 10
    chunk_size = duration / n_chunks
    chunks = []

    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        chunk_words = []
        chunk_speaker_words = defaultdict(list)

        for seg in segments:
            if seg["start"] >= chunk_start and seg["start"] < chunk_end:
                words = tokenize(seg["text"])
                chunk_words.extend(words)
                speaker = seg.get("speaker") or "Unknown"
                chunk_speaker_words[speaker].extend(words)

        chunk_duration_min = chunk_size / 60
        wpm = round(len(chunk_words) / max(chunk_duration_min, 0.1), 1)
        sup_count = sum(chunk_words.count(s) for s in SUPERLATIVES)
        filler_count = sum(chunk_words.count(f) for f in FILLER_WORDS)

        chunk_data = {
            "chunk": i + 1,
            "pct_through": round((i + 0.5) / n_chunks * 100),
            "words_per_minute": wpm,
            "superlatives": sup_count,
            "fillers": filler_count
        }
        # Per-speaker WPM in this chunk
        for speaker in SPEAKERS:
            sw = chunk_speaker_words.get(speaker, [])
            chunk_data[f"wpm_{speaker.lower()}"] = round(len(sw) / max(chunk_duration_min, 0.1), 1)

        chunks.append(chunk_data)

    return chunks


def analyze_reactions(segments):
    full_text = " ".join(s["text"] for s in segments)
    results = {}
    for name, pattern in REACTION_PATTERNS.items():
        results[name] = len(pattern.findall(full_text))
    return results


def analyze_movie_mentions(segments):
    text = " ".join(s["text"] for s in segments)
    title_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    potential_titles = title_pattern.findall(text)
    title_counts = Counter(potential_titles)
    noise = {"Movie Awards", "Justin Tyler", "Tyler Movie", "Justin Movie",
             "Best Movie", "Worst Movie", "Movie Award", "Part One", "Part Two",
             "Part Three", "Best Movies", "Best Director", "Best Actress",
             "Best Actor", "Fan Choice"}
    filtered = {k: v for k, v in title_counts.items() if k not in noise and v >= 1}
    return {
        "total_mentions": sum(filtered.values()),
        "unique_titles": len(filtered),
        "top_mentioned": Counter(filtered).most_common(15)
    }


def analyze_castle(segments):
    """Track mentions of Castle the Rabbit (Tyler's rabbit, named after The Punisher).
    Also detects 'Castle Award' references - an actual award category in the show."""
    mentions = []
    full_text = " ".join(s["text"] for s in segments)

    for seg in segments:
        if CASTLE_PATTERNS.search(seg["text"]):
            mentions.append({
                "time": seg["start"],
                "text": seg["text"].strip(),
            })

    # Count from full text for accuracy
    total_count = len(CASTLE_PATTERNS.findall(full_text))
    castle_award_count = len(re.findall(r'castle\s+award', full_text, re.I))
    movie_refs = len(re.findall(r'infinity\s+castle', full_text, re.I))

    return {
        "total_mentions": total_count,
        "castle_award_mentions": castle_award_count,
        "movie_castle_mentions": movie_refs,
        "rabbit_mentions": max(total_count - castle_award_count - movie_refs, 0),
        "contexts": mentions[:10],
    }


def analyze_sentiment(segments, duration_seconds):
    """Sentiment analysis using VADER. Returns overall scores, timeline, and extreme moments."""
    sia = SentimentIntensityAnalyzer()
    if not segments:
        return {"compound": 0, "pos": 0, "neg": 0, "neu": 0, "timeline": [], "most_positive": None, "most_negative": None}

    # Overall scores
    full_text = " ".join(s["text"] for s in segments)
    overall = sia.polarity_scores(full_text)

    # 10-chunk timeline
    duration = max(duration_seconds, 1)
    n_chunks = 10
    chunk_size = duration / n_chunks
    timeline = []
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        chunk_text = " ".join(s["text"] for s in segments if chunk_start <= s["start"] < chunk_end)
        if chunk_text.strip():
            scores = sia.polarity_scores(chunk_text)
            timeline.append({
                "chunk": i + 1,
                "pct_through": round((i + 0.5) / n_chunks * 100),
                "compound": round(scores["compound"], 3),
                "pos": round(scores["pos"], 3),
                "neg": round(scores["neg"], 3),
            })
        else:
            timeline.append({"chunk": i + 1, "pct_through": round((i + 0.5) / n_chunks * 100), "compound": 0, "pos": 0, "neg": 0})

    # Most positive and negative moments (per-segment)
    seg_scores = []
    for s in segments:
        if s["text"].strip():
            score = sia.polarity_scores(s["text"])
            seg_scores.append({"start": s["start"], "text": s["text"], "compound": score["compound"]})

    most_positive = max(seg_scores, key=lambda x: x["compound"]) if seg_scores else None
    most_negative = min(seg_scores, key=lambda x: x["compound"]) if seg_scores else None

    return {
        "compound": round(overall["compound"], 3),
        "pos": round(overall["pos"], 3),
        "neg": round(overall["neg"], 3),
        "neu": round(overall["neu"], 3),
        "timeline": timeline,
        "most_positive": most_positive,
        "most_negative": most_negative,
    }


def analyze_rants(segments, duration_seconds, sentiment_data):
    """Detect rants using 30-second sliding window with composite score."""
    if not segments or duration_seconds <= 0:
        return {"rant_index": 0, "top_rants": []}

    sia = SentimentIntensityAnalyzer()
    window_size = 30  # seconds
    step = 10  # seconds
    duration = max(duration_seconds, 1)

    windows = []
    t = 0
    while t + window_size <= duration + step:
        window_end = min(t + window_size, duration)
        window_segs = [s for s in segments if t <= s["start"] < window_end]
        window_text = " ".join(s["text"] for s in window_segs)
        words = tokenize(window_text)
        word_count = len(words)

        if word_count < 5:
            t += step
            continue

        # WPM for this window
        window_dur_min = window_size / 60
        wpm = word_count / window_dur_min

        # Negative sentiment
        sentiment = sia.polarity_scores(window_text)
        neg_score = sentiment["neg"]

        # Superlative density
        sup_count = sum(words.count(s) for s in SUPERLATIVES)
        sup_density = sup_count / max(word_count, 1) * 100

        # Filler density
        filler_count = sum(words.count(f) for f in FILLER_WORDS)
        filler_density = filler_count / max(word_count, 1) * 100

        windows.append({
            "start": t,
            "end": window_end,
            "wpm": wpm,
            "neg": neg_score,
            "sup_density": sup_density,
            "filler_density": filler_density,
            "text_preview": window_text[:200],
            "word_count": word_count,
        })
        t += step

    if not windows:
        return {"rant_index": 0, "top_rants": []}

    # Z-score normalization for composite score
    import numpy as np
    wpms = [w["wpm"] for w in windows]
    negs = [w["neg"] for w in windows]
    sups = [w["sup_density"] for w in windows]
    fillers = [w["filler_density"] for w in windows]

    def z_scores(values):
        arr = np.array(values, dtype=float)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std

    wpm_z = z_scores(wpms)
    neg_z = z_scores(negs)
    sup_z = z_scores(sups)
    filler_z = z_scores(fillers)

    for i, w in enumerate(windows):
        w["rant_score"] = round(
            0.35 * wpm_z[i] + 0.30 * neg_z[i] + 0.20 * sup_z[i] + 0.15 * filler_z[i], 3
        )

    windows.sort(key=lambda x: -x["rant_score"])
    top_rants = []
    for w in windows[:5]:
        top_rants.append({
            "start": w["start"],
            "end": w["end"],
            "score": w["rant_score"],
            "wpm": round(w["wpm"], 1),
            "neg": round(w["neg"], 3),
            "sup_density": round(w["sup_density"], 2),
            "filler_density": round(w["filler_density"], 2),
            "text_preview": w["text_preview"],
        })

    rant_index = round(windows[0]["rant_score"], 3) if windows else 0

    return {"rant_index": rant_index, "top_rants": top_rants}


def compute_greatest_rants(all_video_analyses):
    """Find the top 10 greatest rants across all videos."""
    all_rants = []
    for vid_id, va in all_video_analyses.items():
        rants = va.get("rants", {}).get("top_rants", [])
        for r in rants:
            all_rants.append({**r, "video_id": vid_id, "title": va["title"], "year": va["year"]})
    all_rants.sort(key=lambda x: -x["score"])
    return all_rants[:10]


def generate_llm_quotes(sample_text):
    """Generate fake movie critic quotes using Google Gemini Flash."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("  GOOGLE_API_KEY not set, skipping LLM quotes")
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = f"""You are a comedy writer. Two brothers named Justin and Tyler run a YouTube movie awards show.
They're passionate, opinionated, and love hyperbole. Based on how they actually talk (samples below),
generate 20 funny fake quotes that sound like they COULD have said them but didn't.

Rules:
- Each quote should be 1-2 sentences, punchy and quotable
- Mix of positive and negative opinions about movies
- Include their verbal tics: "literally", "honestly", "like", "insane"
- Some should be absurd hot takes, some should be weirdly profound
- Don't reference specific real movies — keep them generic enough to be timeless
- Format: return ONLY a JSON array of 20 strings, no other text

Sample transcript excerpts:
{sample_text[:3000]}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = response.text.strip()
        # Extract JSON array from response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        log(f"  Gemini quote generation failed: {e}")
        return None


def build_markov_chain(all_segments, order=2):
    """Build a 2nd-order Markov chain and generate fake quotes (fallback if LLM unavailable)."""
    # Build chain from all text
    chain = defaultdict(list)
    for seg in all_segments:
        words = seg["text"].split()
        for i in range(len(words) - order):
            key = tuple(words[i:i + order])
            chain[key].append(words[i + order])

    if not chain:
        return []

    # Generate quotes with seed for reproducibility
    rng = random.Random(42)
    all_text = " ".join(s["text"] for s in all_segments).lower()
    quotes = []
    keys = list(chain.keys())
    attempts = 0

    while len(quotes) < 20 and attempts < 200:
        attempts += 1
        # Pick a random starting key
        key = rng.choice(keys)
        words = list(key)
        target_len = rng.randint(15, 40)

        for _ in range(target_len - order):
            current_key = tuple(words[-order:])
            if current_key not in chain:
                break
            next_word = rng.choice(chain[current_key])
            words.append(next_word)

        if len(words) < 15:
            continue

        quote = " ".join(words)
        # Filter out exact substring matches
        if quote.lower() in all_text:
            continue
        quotes.append(quote)

    return quotes


def compute_tfidf_fingerprints(all_video_data):
    """Compute TF-IDF signatures per year and track specific words over time."""
    year_texts = defaultdict(list)
    for vid_id, vdata in all_video_data.items():
        year = vdata.get("year")
        if not year:
            continue
        text = " ".join(s["text"] for s in vdata["segments"])
        year_texts[year].append(text)

    if not year_texts:
        return {"year_signatures": {}, "word_trends": {}}

    # One document per year
    years_sorted = sorted(year_texts.keys())
    documents = [" ".join(year_texts[y]) for y in years_sorted]

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", min_df=1, max_df=0.9,
        token_pattern=r'\b[a-z]{3,}\b'
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Top 10 signature words per year
    year_signatures = {}
    for i, year in enumerate(years_sorted):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-10:][::-1]
        year_signatures[year] = [
            {"word": feature_names[idx], "score": round(float(row[idx]), 4)}
            for idx in top_indices if row[idx] > 0
        ]

    # Track specific words across years
    tracked_words = ["literally", "amazing", "incredible", "masterpiece", "terrible", "insane", "honestly"]
    word_trends = {}
    for word in tracked_words:
        if word in feature_names:
            word_idx = list(feature_names).index(word)
            trend = {}
            for i, year in enumerate(years_sorted):
                score = float(tfidf_matrix[i, word_idx])
                if score > 0:
                    trend[year] = round(score, 4)
            if trend:
                word_trends[word] = trend

    return {"year_signatures": year_signatures, "word_trends": word_trends}


def train_castle_predictor(all_video_analyses):
    """Train logistic regression to predict Castle mentions."""
    features = []
    targets = []
    video_ids = []

    for vid_id, va in all_video_analyses.items():
        castle = va.get("castle", {})
        rabbit_mentions = castle.get("rabbit_mentions", 0)
        vocab = va.get("vocabulary", {})
        sups = va.get("superlatives", {})
        fillers = va.get("filler_words", {})
        hot_takes = va.get("hot_takes", {})

        total_words = vocab.get("total_words", 0)
        if total_words == 0:
            continue

        duration_min = va.get("duration_min", 1) or 1
        total_filler = sum(f["count"] for f in fillers.values()) if isinstance(fillers, dict) else 0
        filler_density = total_filler / max(total_words, 1) * 100

        feature_row = [
            float(va.get("year", 2020) or 2020),
            float(duration_min),
            float(total_words),
            float(vocab.get("words_per_minute", 0) or 0),
            float(filler_density),
            float(sups.get("hyperbole_index", 0) or 0),
            float(hot_takes.get("count", 0) or 0),
        ]
        # Skip rows with any NaN/None
        if any(v != v for v in feature_row):
            continue
        features.append(feature_row)
        targets.append(1 if rabbit_mentions > 0 else 0)
        video_ids.append(vid_id)

    if len(features) < 10 or sum(targets) < 2:
        return {"error": "Not enough data for training", "coefficients": [], "accuracy": 0, "probabilities": {}}

    import numpy as np
    X = np.array(features)
    y = np.array(targets)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    feature_names = ["year", "duration_min", "word_count", "wpm", "filler_density", "hyperbole_index", "hot_takes"]
    coefficients = [
        {"feature": name, "coefficient": round(float(coef), 4)}
        for name, coef in zip(feature_names, model.coef_[0])
    ]
    coefficients.sort(key=lambda x: -abs(x["coefficient"]))

    accuracy = round(float(model.score(X_scaled, y)), 3)

    # Per-video probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    probabilities = {video_ids[i]: round(float(probs[i]), 3) for i in range(len(video_ids))}

    return {
        "coefficients": coefficients,
        "accuracy": accuracy,
        "probabilities": probabilities,
        "n_positive": int(sum(targets)),
        "n_total": len(targets),
    }


def analyze_video(vdata):
    """Run all analyses for a single video, including per-speaker breakdowns."""
    segments = vdata["segments"]
    if not segments:
        return None

    duration = vdata["duration"]

    result = {
        "video_id": vdata["video_id"],
        "title": vdata["title"],
        "year": vdata["year"],
        "category": vdata["category"],
        "duration_min": round(duration / 60, 1),
        "speaker_times": vdata.get("speaker_times", {}),
        # Aggregate analysis (both speakers combined)
        "word_frequency": analyze_word_frequency(segments),
        "filler_words": analyze_filler_words(segments, duration),
        "superlatives": analyze_superlatives(segments, duration),
        "agreement_disagreement": analyze_agreement_disagreement(segments),
        "hot_takes": analyze_hot_takes(segments),
        "vocabulary": analyze_vocabulary_richness(segments, duration),
        "energy_timeline": analyze_energy_over_time(vdata),
        "reactions": analyze_reactions(segments),
        "movie_mentions": analyze_movie_mentions(segments),
        "castle": analyze_castle(segments),
        "sentiment": analyze_sentiment(segments, duration),
    }

    # Rant detection (depends on sentiment)
    result["rants"] = analyze_rants(segments, duration, result["sentiment"])

    # Per-speaker analysis (if diarization data exists)
    has_speaker_data = any(s.get("speaker") for s in segments)
    result["has_speaker_data"] = has_speaker_data

    if has_speaker_data:
        by_speaker = {}
        for speaker in SPEAKERS:
            sp_segments = get_segments_for_speaker(vdata, speaker)
            if not sp_segments:
                continue
            # Estimate speaker's duration from their segment count ratio
            sp_duration = sum(s["duration"] for s in sp_segments)
            sp_duration_total = max(vdata.get("speaker_times", {}).get(speaker, sp_duration), 1)

            by_speaker[speaker] = {
                "segment_count": len(sp_segments),
                "speaking_time_seconds": round(sp_duration_total, 1),
                "word_frequency": analyze_word_frequency(sp_segments),
                "filler_words": analyze_filler_words(sp_segments, sp_duration_total),
                "superlatives": analyze_superlatives(sp_segments, sp_duration_total),
                "vocabulary": analyze_vocabulary_richness(sp_segments, sp_duration_total),
                "hot_takes": analyze_hot_takes(sp_segments),
                "reactions": analyze_reactions(sp_segments),
            }
        result["by_speaker"] = by_speaker

    return result


def compute_speaker_aggregates(all_data):
    """Compute aggregate per-speaker stats across all videos."""
    speaker_all_words = defaultdict(list)
    speaker_total_time = defaultdict(float)
    speaker_filler_counts = defaultdict(Counter)
    speaker_superlative_counts = defaultdict(Counter)
    speaker_hot_takes = defaultdict(int)
    speaker_video_count = defaultdict(int)

    for vid_id, vdata in all_data.items():
        has_speaker = any(s.get("speaker") for s in vdata["segments"])
        if not has_speaker:
            continue

        for speaker in SPEAKERS:
            sp_segs = get_segments_for_speaker(vdata, speaker)
            if not sp_segs:
                continue

            speaker_video_count[speaker] += 1
            sp_time = vdata.get("speaker_times", {}).get(speaker, sum(s["duration"] for s in sp_segs))
            speaker_total_time[speaker] += sp_time

            text = text_from_segments(sp_segs)
            words = tokenize(text)
            speaker_all_words[speaker].extend(words)

            for f in FILLER_WORDS:
                speaker_filler_counts[speaker][f] += words.count(f)
            for s in SUPERLATIVES:
                speaker_superlative_counts[speaker][s] += words.count(s)
            speaker_hot_takes[speaker] += analyze_hot_takes(sp_segs)["count"]

    result = {}
    for speaker in SPEAKERS:
        words = speaker_all_words[speaker]
        if not words:
            continue
        duration_min = max(speaker_total_time[speaker] / 60, 1)
        filtered = [w for w in words if w not in STOP_WORDS and w not in EXTRA_STOPS and len(w) > 2]
        unique = set(words)

        result[speaker] = {
            "total_words": len(words),
            "unique_words": len(unique),
            "ttr": round(len(unique) / max(len(words), 1), 4),
            "words_per_minute": round(len(words) / duration_min, 1),
            "total_speaking_time_min": round(speaker_total_time[speaker] / 60, 1),
            "videos_with_data": speaker_video_count[speaker],
            "top_words": Counter(filtered).most_common(20),
            "top_fillers": speaker_filler_counts[speaker].most_common(10),
            "total_fillers": sum(speaker_filler_counts[speaker].values()),
            "filler_per_minute": round(sum(speaker_filler_counts[speaker].values()) / duration_min, 2),
            "top_superlatives": speaker_superlative_counts[speaker].most_common(10),
            "total_superlatives": sum(speaker_superlative_counts[speaker].values()),
            "superlative_per_minute": round(sum(speaker_superlative_counts[speaker].values()) / duration_min, 2),
            "hot_takes": speaker_hot_takes[speaker],
        }

    return result


def compute_year_aggregates(all_data):
    """Aggregate stats by year, including per-speaker breakdowns."""
    year_groups = defaultdict(list)
    for vid_id, vdata in all_data.items():
        if vdata["year"]:
            year_groups[vdata["year"]].append(vdata)

    year_stats = {}
    for year, videos in sorted(year_groups.items()):
        all_words = []
        total_duration = 0
        total_fillers = Counter()
        total_superlatives = Counter()
        total_agreements = 0
        total_disagreements = 0
        total_hot_takes = 0

        # Per-speaker year tracking
        sp_words = defaultdict(list)
        sp_time = defaultdict(float)
        sp_fillers = defaultdict(Counter)
        sp_superlatives = defaultdict(Counter)

        for v in videos:
            segments = v["segments"]
            text = text_from_segments(segments)
            words = tokenize(text)
            all_words.extend(words)
            total_duration += v["duration"]

            for f in FILLER_WORDS:
                total_fillers[f] += words.count(f)
            for s in SUPERLATIVES:
                total_superlatives[s] += words.count(s)
            ad = analyze_agreement_disagreement(segments)
            total_agreements += ad["agreements"]
            total_disagreements += ad["disagreements"]
            total_hot_takes += analyze_hot_takes(segments)["count"]

            # Per-speaker
            for speaker in SPEAKERS:
                sp_segs = get_segments_for_speaker(v, speaker)
                if sp_segs:
                    sp_text = text_from_segments(sp_segs)
                    sw = tokenize(sp_text)
                    sp_words[speaker].extend(sw)
                    sp_t = v.get("speaker_times", {}).get(speaker, sum(s["duration"] for s in sp_segs))
                    sp_time[speaker] += sp_t
                    for f in FILLER_WORDS:
                        sp_fillers[speaker][f] += sw.count(f)
                    for s in SUPERLATIVES:
                        sp_superlatives[speaker][s] += sw.count(s)

        duration_min = max(total_duration / 60, 1)
        unique_words = set(all_words)
        filtered_words = [w for w in all_words if w not in STOP_WORDS and w not in EXTRA_STOPS and len(w) > 2]

        year_stat = {
            "year": year,
            "num_videos": len(videos),
            "total_duration_min": round(total_duration / 60, 1),
            "total_words": len(all_words),
            "unique_words": len(unique_words),
            "ttr": round(len(unique_words) / max(len(all_words), 1), 4),
            "words_per_minute": round(len(all_words) / duration_min, 1),
            "top_words": Counter(filtered_words).most_common(20),
            "top_fillers": total_fillers.most_common(10),
            "top_superlatives": total_superlatives.most_common(10),
            "total_filler_count": sum(total_fillers.values()),
            "filler_per_minute": round(sum(total_fillers.values()) / duration_min, 2),
            "superlative_per_minute": round(sum(total_superlatives.values()) / duration_min, 2),
            "agreements": total_agreements,
            "disagreements": total_disagreements,
            "harmony_ratio": round(total_agreements / max(total_disagreements, 1), 2),
            "hot_takes": total_hot_takes,
            "categories": list(set(v["category"] for v in videos)),
        }

        # Add per-speaker year data
        by_speaker_year = {}
        for speaker in SPEAKERS:
            sw = sp_words[speaker]
            if sw:
                sp_dur = max(sp_time[speaker] / 60, 1)
                sp_unique = set(sw)
                by_speaker_year[speaker] = {
                    "total_words": len(sw),
                    "words_per_minute": round(len(sw) / sp_dur, 1),
                    "ttr": round(len(sp_unique) / max(len(sw), 1), 4),
                    "filler_per_minute": round(sum(sp_fillers[speaker].values()) / sp_dur, 2),
                    "superlative_per_minute": round(sum(sp_superlatives[speaker].values()) / sp_dur, 2),
                    "speaking_time_min": round(sp_time[speaker] / 60, 1),
                }
        year_stat["by_speaker"] = by_speaker_year
        year_stats[year] = year_stat

    return year_stats


def compute_all_time_stats(all_data):
    """Fun all-time records and stats."""
    records = {
        "longest_video": None,
        "most_words_video": None,
        "highest_wpm_video": None,
        "most_fillers_video": None,
        "most_superlatives_video": None,
        "richest_vocabulary_video": None,
        "most_hot_takes_video": None,
        "most_harmonious_video": None,
        "most_combative_video": None,
    }

    longest_dur = 0
    most_words = 0
    highest_wpm = 0
    most_fillers = 0
    most_sups = 0
    richest_ttr = 0
    most_ht = 0
    most_harmony = 0
    most_combat = 0

    for vid_id, vdata in all_data.items():
        if not vdata["segments"]:
            continue

        text = text_from_segments(vdata["segments"])
        words = tokenize(text)
        duration_min = max(vdata["duration"] / 60, 1)
        wpm = len(words) / duration_min

        filler_count = sum(words.count(f) for f in FILLER_WORDS)
        sup_count = sum(words.count(s) for s in SUPERLATIVES)
        ttr = len(set(words)) / max(len(words), 1)
        ht = analyze_hot_takes(vdata["segments"])["count"]
        ad = analyze_agreement_disagreement(vdata["segments"])

        if vdata["duration"] > longest_dur:
            longest_dur = vdata["duration"]
            records["longest_video"] = {"title": vdata["title"], "duration_min": round(vdata["duration"]/60, 1), "year": vdata["year"]}

        if len(words) > most_words:
            most_words = len(words)
            records["most_words_video"] = {"title": vdata["title"], "word_count": len(words), "year": vdata["year"]}

        if wpm > highest_wpm and len(words) > 100:
            highest_wpm = wpm
            records["highest_wpm_video"] = {"title": vdata["title"], "wpm": round(wpm, 1), "year": vdata["year"]}

        if filler_count > most_fillers:
            most_fillers = filler_count
            records["most_fillers_video"] = {"title": vdata["title"], "filler_count": filler_count, "year": vdata["year"]}

        if sup_count > most_sups:
            most_sups = sup_count
            records["most_superlatives_video"] = {"title": vdata["title"], "superlative_count": sup_count, "year": vdata["year"]}

        if ttr > richest_ttr and len(words) > 200:
            richest_ttr = ttr
            records["richest_vocabulary_video"] = {"title": vdata["title"], "ttr": round(ttr, 4), "year": vdata["year"]}

        if ht > most_ht:
            most_ht = ht
            records["most_hot_takes_video"] = {"title": vdata["title"], "hot_takes": ht, "year": vdata["year"]}

        if ad["harmony_ratio"] > most_harmony and ad["total_opinions"] > 2:
            most_harmony = ad["harmony_ratio"]
            records["most_harmonious_video"] = {"title": vdata["title"], "ratio": ad["harmony_ratio"], "year": vdata["year"]}

        if ad["disagreements"] > most_combat:
            most_combat = ad["disagreements"]
            records["most_combative_video"] = {"title": vdata["title"], "disagreements": ad["disagreements"], "year": vdata["year"]}

    return records


def main():
    log("Connecting to Turso...")
    conn = get_db()

    log("Loading data...")
    all_data = load_all_data(conn)
    log(f"Loaded {len(all_data)} videos")

    if not all_data:
        log("No data found! Run extract_data.py first.")
        sys.exit(1)

    # Identify videos without transcripts
    missing_videos = []
    for vid_id, vdata in all_data.items():
        if not vdata["segments"]:
            missing_videos.append({"video_id": vid_id, "title": vdata["title"], "year": vdata["year"]})

    # Check diarization coverage
    diarized = sum(1 for v in all_data.values() if v.get("speaker_times"))
    log(f"Speaker diarization available for {diarized}/{len(all_data)} videos")

    # Per-video analysis
    log("Analyzing individual videos...")
    video_analyses = {}
    for vid_id, vdata in all_data.items():
        result = analyze_video(vdata)
        if result:
            video_analyses[vid_id] = result

    # Year aggregates
    log("Computing year aggregates...")
    year_stats = compute_year_aggregates(all_data)

    # Speaker aggregates
    log("Computing speaker aggregates...")
    speaker_stats = compute_speaker_aggregates(all_data)

    # All-time records
    log("Finding all-time records...")
    all_time = compute_all_time_stats(all_data)

    # Sentiment year aggregates
    log("Computing sentiment aggregates...")
    sentiment_by_year = {}
    for year, ydata in year_stats.items():
        year_videos = [va for va in video_analyses.values() if va.get("year") == year]
        if year_videos:
            compounds = [v["sentiment"]["compound"] for v in year_videos if v.get("sentiment")]
            if compounds:
                avg_compound = round(sum(compounds) / len(compounds), 3)
                angriest = min(year_videos, key=lambda v: v.get("sentiment", {}).get("compound", 0))
                most_wholesome = max(year_videos, key=lambda v: v.get("sentiment", {}).get("compound", 0))
                sentiment_by_year[year] = {
                    "avg_compound": avg_compound,
                    "angriest_video": {"title": angriest["title"], "compound": angriest["sentiment"]["compound"]},
                    "most_wholesome_video": {"title": most_wholesome["title"], "compound": most_wholesome["sentiment"]["compound"]},
                }

    # All-time sentiment records
    all_sentiments = [(va["title"], va["year"], va["sentiment"]["compound"])
                      for va in video_analyses.values() if va.get("sentiment")]
    angriest_video = min(all_sentiments, key=lambda x: x[2]) if all_sentiments else None
    most_wholesome_video = max(all_sentiments, key=lambda x: x[2]) if all_sentiments else None

    # Greatest rants
    log("Finding greatest rants...")
    greatest_rants = compute_greatest_rants(video_analyses)

    # TF-IDF fingerprints
    log("Computing TF-IDF fingerprints...")
    tfidf = compute_tfidf_fingerprints(all_data)

    # AI-generated quotes (Gemini Flash, with Markov fallback)
    log("Generating AI movie critic quotes...")
    all_segments_flat = []
    for vdata in all_data.values():
        all_segments_flat.extend(vdata["segments"])
    sample_text = " ".join(s["text"] for s in all_segments_flat[:200])
    llm_quotes = generate_llm_quotes(sample_text)
    if llm_quotes:
        markov_quotes = llm_quotes
        quote_source = "gemini"
        log(f"  Generated {len(markov_quotes)} quotes via Gemini Flash")
    else:
        markov_quotes = build_markov_chain(all_segments_flat, order=2)
        quote_source = "markov"
        log(f"  Generated {len(markov_quotes)} quotes via Markov chain (fallback)")

    # Castle predictor
    log("Training Castle mention predictor...")
    castle_predictor = train_castle_predictor(video_analyses)
    if castle_predictor.get("accuracy"):
        log(f"  Predictor accuracy: {castle_predictor['accuracy']}")

    # Castle the Rabbit aggregate stats
    log("Tracking Castle the Rabbit...")
    castle_total = 0
    castle_award_total = 0
    castle_by_year = {}
    castle_top_videos = []
    for vid_id, va in video_analyses.items():
        c = va.get("castle", {})
        count = c.get("total_mentions", 0)
        castle_total += count
        castle_award_total += c.get("castle_award_mentions", 0)
        if count > 0:
            castle_top_videos.append({"title": va["title"], "year": va["year"], "count": count})
        yr = va.get("year")
        if yr:
            castle_by_year[yr] = castle_by_year.get(yr, 0) + count
    castle_top_videos.sort(key=lambda x: -x["count"])

    castle_stats = {
        "total_mentions": castle_total,
        "castle_award_mentions": castle_award_total,
        "by_year": castle_by_year,
        "top_videos": castle_top_videos[:10],
    }
    log(f"  Castle mentioned {castle_total} times across all videos")

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_videos": len(all_data),
        "total_with_transcripts": len(video_analyses),
        "diarized_videos": diarized,
        "has_speaker_data": diarized > 0,
        "videos": video_analyses,
        "by_year": year_stats,
        "by_speaker": speaker_stats,
        "all_time_records": all_time,
        "castle": castle_stats,
        "sentiment_by_year": sentiment_by_year,
        "angriest_video": {"title": angriest_video[0], "year": angriest_video[1], "compound": angriest_video[2]} if angriest_video else None,
        "most_wholesome_video": {"title": most_wholesome_video[0], "year": most_wholesome_video[1], "compound": most_wholesome_video[2]} if most_wholesome_video else None,
        "greatest_rants": greatest_rants,
        "tfidf": tfidf,
        "markov_quotes": markov_quotes,
        "quote_source": quote_source,
        "castle_predictor": castle_predictor,
        "missing_videos": missing_videos,
    }

    output_path = os.path.join(os.path.dirname(__file__), "public", "data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nAnalysis written to {output_path}")
    log(f"  {len(video_analyses)} videos analyzed")
    log(f"  {len(year_stats)} years covered")
    log(f"  {diarized} videos with speaker data")
    if speaker_stats:
        for sp, stats in speaker_stats.items():
            log(f"  {sp}: {stats['total_words']} words, {stats['total_speaking_time_min']} min")


if __name__ == "__main__":
    main()
