#!/usr/bin/env python3
"""
Extract video metadata and transcripts for all "movie awards" videos
from @BridgewatersFinest YouTube channel, then store in Turso.

Usage:
    python extract_data.py          # Incremental: only fetch new videos
    python extract_data.py --full   # Full refresh: re-fetch everything
"""
import argparse
import os
import re
import subprocess
from datetime import datetime, timezone

import libsql_experimental as libsql
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

CHANNEL_URL = "https://www.youtube.com/@BridgewatersFinest/videos"
TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")


def log(msg):
    print(msg, flush=True)


def get_db():
    return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)


def create_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            duration_seconds REAL,
            year INTEGER,
            category TEXT,
            upload_date TEXT
        );

        CREATE TABLE IF NOT EXISTS transcript_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            duration REAL NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            videos_added INTEGER DEFAULT 0,
            transcripts_added INTEGER DEFAULT 0
        );
    """)
    # Migration: add upload_date column if missing
    try:
        conn.execute("ALTER TABLE videos ADD COLUMN upload_date TEXT")
    except Exception:
        pass  # Column already exists
    conn.commit()


def get_existing_video_ids(conn):
    rows = conn.execute("SELECT video_id FROM videos").fetchall()
    return {row[0] for row in rows}


def fetch_channel_videos():
    """Use yt-dlp to get all video IDs, titles, durations, and upload dates."""
    log("Fetching channel video list from YouTube...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist",
         "--print", "%(id)s|||%(title)s|||%(duration)s|||%(upload_date)s",
         CHANNEL_URL],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        log(f"yt-dlp error: {result.stderr[:500]}")
    videos = []
    for line in result.stdout.strip().split("\n"):
        if "|||" not in line:
            continue
        parts = line.split("|||")
        if len(parts) >= 3:
            vid_id = parts[0].strip()
            title = parts[1].strip()
            dur = parts[2].strip()
            upload_date = parts[3].strip() if len(parts) > 3 else "NA"
            videos.append({
                "video_id": vid_id,
                "title": title,
                "duration": float(dur) if dur not in ("NA", "") else 0,
                "upload_date": upload_date if upload_date != "NA" else None
            })
    return videos


def filter_movie_awards(videos):
    return [v for v in videos if "movie awards" in v["title"].lower()]


def parse_year(title):
    year_match = re.search(r'(?:of |Movies of )\s*(20\d{2})', title)
    if year_match:
        return int(year_match.group(1))
    year_match = re.search(r'(20\d{2})', title)
    if year_match:
        return int(year_match.group(1))
    return None


def parse_category(title):
    t = title.lower()
    if "opening ceremony" in t or "opening (" in t:
        return "Opening Ceremony"
    if "worst" in t:
        return "Worst Movies"
    if "fan" in t and "choice" in t:
        return "Fan's Choice"
    if any(x in t for x in ["best movies", "top 10", "top 30", "top twenty", "top 20", "top ten"]):
        return "Best Movies"
    if "actress" in t:
        return "Best Actresses"
    if "actor" in t:
        return "Best Actors"
    if "director" in t:
        return "Best Directors"
    if "miscellaneous" in t:
        return "Miscellaneous"
    if "mid-year" in t:
        return "Mid-Year Awards"
    if "retrospective" in t:
        return "Retrospective"
    if "genre" in t:
        return "Best Genre Movies"
    if "expected more" in t or "hated" in t or "loved" in t:
        return "Hated & Loved"
    if "other media" in t:
        return "Other Media"
    if "hall of fame" in t:
        return "Hall of Fame"
    if "after credits" in t:
        return "After Credits"
    if "honourable mention" in t:
        return "Honourable Mentions"
    return "Other"


def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        return [
            {"start": s.start, "duration": s.duration, "text": s.text}
            for s in transcript
        ]
    except Exception as e:
        log(f"  WARNING: No transcript for {video_id}: {e}")
        return []


def store_video(conn, video):
    conn.execute(
        "INSERT OR REPLACE INTO videos (video_id, title, duration_seconds, year, category, upload_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (video["video_id"], video["title"], video["duration"],
         parse_year(video["title"]), parse_category(video["title"]),
         video.get("upload_date"))
    )


def store_transcript(conn, video_id, segments):
    conn.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
    for seg in segments:
        conn.execute(
            "INSERT INTO transcript_segments (video_id, start_time, duration, text) VALUES (?, ?, ?, ?)",
            (video_id, seg["start"], seg["duration"], seg["text"])
        )


def main():
    parser = argparse.ArgumentParser(description="Extract movie awards video data")
    parser.add_argument("--full", action="store_true", help="Full refresh (ignore existing data)")
    args = parser.parse_args()

    conn = get_db()
    create_tables(conn)

    existing_ids = set() if args.full else get_existing_video_ids(conn)
    if existing_ids:
        log(f"Found {len(existing_ids)} existing videos in database. Running incremental update.")
    else:
        log("No existing data found. Running full extraction.")

    all_videos = fetch_channel_videos()
    movie_awards = filter_movie_awards(all_videos)
    log(f"Found {len(movie_awards)} movie awards videos on channel")

    new_videos = [v for v in movie_awards if v["video_id"] not in existing_ids]
    if not new_videos and not args.full:
        log("No new videos to process. Database is up to date!")
        return

    log(f"Processing {len(new_videos)} {'total' if args.full else 'new'} videos...")

    videos_added = 0
    transcripts_added = 0

    for i, video in enumerate(new_videos):
        log(f"\n[{i+1}/{len(new_videos)}] {video['title']}")
        store_video(conn, video)
        videos_added += 1

        # Check if we already have transcript segments (for --full mode)
        existing_segs = conn.execute(
            "SELECT COUNT(*) FROM transcript_segments WHERE video_id = ?",
            (video["video_id"],)
        ).fetchone()[0]

        if existing_segs > 0 and not args.full:
            log(f"  Already have {existing_segs} transcript segments, skipping.")
            continue

        segments = fetch_transcript(video["video_id"])
        if segments:
            store_transcript(conn, video["video_id"], segments)
            transcripts_added += 1
            log(f"  Stored {len(segments)} transcript segments")

        conn.commit()

    # Record this pipeline run
    conn.execute(
        "INSERT INTO pipeline_runs (run_date, videos_added, transcripts_added) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), videos_added, transcripts_added)
    )
    conn.commit()
    log(f"\nDone! Added {videos_added} videos, {transcripts_added} transcripts.")


if __name__ == "__main__":
    main()
