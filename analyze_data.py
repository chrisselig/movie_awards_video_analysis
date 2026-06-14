#!/usr/bin/env python3
"""
Analyze movie awards transcripts and generate JSON data for the D3.js infographic.

Supports per-speaker analysis when diarization data is available.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import libsql_experimental as libsql
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords

load_dotenv()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
STOP_WORDS = set(stopwords.words("english"))

EXTRA_STOPS = {"like", "know", "yeah", "okay", "oh", "well", "go", "going",
               "got", "get", "thing", "things", "really", "right", "uh",
               "um", "na", "gon", "wan", "one", "would", "could", "also",
               "think", "much", "good", "even", "way", "lot", "say", "said",
               "see", "still", "actually", "pretty", "make", "take", "come",
               "back", "something", "kind", "mean", "want", "let", "put",
               "look", "two", "first", "next", "number"}

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
    }

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
