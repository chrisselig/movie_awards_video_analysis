#!/usr/bin/env python3
"""
Speaker diarization pipeline for Justin & Tyler Movie Awards videos.

Downloads audio from YouTube, runs pyannote speaker diarization,
and maps speaker labels to transcript segments in Turso.

Prerequisites:
    - ffmpeg installed (sudo apt install ffmpeg)
    - Hugging Face token with access to pyannote models
      Set HF_TOKEN env var or pass --hf-token

Usage:
    python diarize.py                    # Process videos missing diarization
    python diarize.py --video VIDEO_ID   # Process a single video
    python diarize.py --full             # Re-process all videos
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import libsql_experimental as libsql
from dotenv import load_dotenv

load_dotenv()

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN", "")
AUDIO_DIR = Path(__file__).parent / "audio"


def log(msg):
    print(msg, flush=True)


def get_db():
    return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)


def create_diarization_table(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS speaker_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            speaker_label TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            duration REAL NOT NULL
        );
    """)
    # Add speaker column to transcript_segments if missing
    try:
        conn.execute("ALTER TABLE transcript_segments ADD COLUMN speaker TEXT")
    except Exception:
        pass
    conn.commit()


def get_videos_needing_diarization(conn):
    """Find videos that have transcripts but no diarization data."""
    rows = conn.execute("""
        SELECT DISTINCT v.video_id, v.title, v.duration_seconds
        FROM videos v
        JOIN transcript_segments ts ON v.video_id = ts.video_id
        WHERE v.video_id NOT IN (SELECT DISTINCT video_id FROM speaker_segments)
        ORDER BY v.year DESC, v.title
    """).fetchall()
    return [{"video_id": r[0], "title": r[1], "duration": r[2]} for r in rows]


def download_audio(video_id):
    """Download audio from YouTube video using yt-dlp."""
    AUDIO_DIR.mkdir(exist_ok=True)
    output_path = AUDIO_DIR / f"{video_id}.wav"

    if output_path.exists():
        log(f"  Audio already downloaded: {output_path}")
        return output_path

    log("  Downloading audio...")
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "5",
         "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
         "-o", str(AUDIO_DIR / "%(id)s.%(ext)s"),
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        log(f"  ERROR downloading audio: {result.stderr[:300]}")
        return None

    if output_path.exists():
        return output_path

    # yt-dlp might save as .webm first, check
    webm_path = AUDIO_DIR / f"{video_id}.webm"
    if webm_path.exists():
        subprocess.run(
            ["ffmpeg", "-i", str(webm_path), "-ac", "1", "-ar", "16000",
             str(output_path)],
            capture_output=True, timeout=300
        )
        webm_path.unlink()
        if output_path.exists():
            return output_path

    log("  ERROR: Audio file not found after download")
    return None


def run_diarization(audio_path, hf_token):
    """Run pyannote speaker diarization on audio file."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log("  ERROR: pyannote.audio not installed. Run: pip install pyannote.audio")
        return None

    log("  Running diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Run diarization
    diarization = pipeline(str(audio_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.end - turn.start
        })

    return segments


def assign_speaker_names(segments):
    """
    Map generic speaker labels (SPEAKER_00, SPEAKER_01) to Justin/Tyler.
    Heuristic: In most videos, the person who speaks first in the opening
    is Justin (he typically starts the show). We assign based on total
    speaking time - the host who talks more is likely Justin as the
    primary presenter in many segments.

    This is a heuristic and may need manual correction for some videos.
    """
    if not segments:
        return segments

    speaker_times = {}
    for seg in segments:
        sp = seg["speaker"]
        speaker_times[sp] = speaker_times.get(sp, 0) + seg["duration"]

    # Sort speakers by who appears first
    first_appearances = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in first_appearances:
            first_appearances[sp] = seg["start"]

    speakers_by_first = sorted(first_appearances.items(), key=lambda x: x[1])

    # The first speaker to appear is labeled as Justin (he usually opens)
    name_map = {}
    if len(speakers_by_first) >= 2:
        name_map[speakers_by_first[0][0]] = "Justin"
        name_map[speakers_by_first[1][0]] = "Tyler"
    elif len(speakers_by_first) == 1:
        name_map[speakers_by_first[0][0]] = "Justin"

    # Map any remaining speakers as "Other"
    for seg in segments:
        if seg["speaker"] not in name_map:
            name_map[seg["speaker"]] = "Other"

    for seg in segments:
        seg["speaker_name"] = name_map.get(seg["speaker"], "Unknown")

    return segments


def store_diarization(conn, video_id, segments):
    """Store diarization results and update transcript segments."""
    conn.execute("DELETE FROM speaker_segments WHERE video_id = ?", (video_id,))

    for seg in segments:
        conn.execute(
            "INSERT INTO speaker_segments (video_id, speaker_label, start_time, end_time, duration) "
            "VALUES (?, ?, ?, ?, ?)",
            (video_id, seg.get("speaker_name", seg["speaker"]),
             seg["start"], seg["end"], seg["duration"])
        )

    # Map transcript segments to speakers based on timestamp overlap
    transcript_segs = conn.execute(
        "SELECT id, start_time, duration FROM transcript_segments WHERE video_id = ?",
        (video_id,)
    ).fetchall()

    for ts_id, ts_start, ts_dur in transcript_segs:
        ts_mid = ts_start + ts_dur / 2  # Use midpoint for matching
        best_speaker = None
        for seg in segments:
            if seg["start"] <= ts_mid <= seg["end"]:
                best_speaker = seg.get("speaker_name", seg["speaker"])
                break
        if best_speaker:
            conn.execute(
                "UPDATE transcript_segments SET speaker = ? WHERE id = ?",
                (best_speaker, ts_id)
            )

    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization for movie awards videos")
    parser.add_argument("--video", help="Process a single video by ID")
    parser.add_argument("--full", action="store_true", help="Re-process all videos")
    parser.add_argument("--hf-token", default=HF_TOKEN, help="Hugging Face auth token")
    parser.add_argument("--cleanup-audio", action="store_true",
                        help="Delete audio files after processing")
    args = parser.parse_args()

    if not args.hf_token:
        log("ERROR: Hugging Face token required. Set HF_TOKEN env var or use --hf-token")
        log("Get a token at https://huggingface.co/settings/tokens")
        log("You need access to pyannote/speaker-diarization-3.1:")
        log("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        log("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)

    conn = get_db()
    create_diarization_table(conn)

    if args.video:
        videos = [{"video_id": args.video, "title": args.video, "duration": 0}]
    elif args.full:
        videos = [{"video_id": r[0], "title": r[1], "duration": r[2]}
                  for r in conn.execute(
                      "SELECT DISTINCT v.video_id, v.title, v.duration_seconds "
                      "FROM videos v JOIN transcript_segments ts ON v.video_id = ts.video_id"
                  ).fetchall()]
    else:
        videos = get_videos_needing_diarization(conn)

    if not videos:
        log("No videos need diarization. Use --full to re-process all.")
        return

    log(f"Processing {len(videos)} videos for speaker diarization...")

    for i, video in enumerate(videos):
        log(f"\n[{i+1}/{len(videos)}] {video['title']}")

        audio_path = download_audio(video["video_id"])
        if not audio_path:
            continue

        segments = run_diarization(audio_path, args.hf_token)
        if not segments:
            continue

        segments = assign_speaker_names(segments)
        store_diarization(conn, video["video_id"], segments)

        speaker_summary = {}
        for seg in segments:
            name = seg.get("speaker_name", seg["speaker"])
            speaker_summary[name] = speaker_summary.get(name, 0) + seg["duration"]

        for name, dur in sorted(speaker_summary.items(), key=lambda x: -x[1]):
            log(f"  {name}: {dur/60:.1f} minutes")

        if args.cleanup_audio and audio_path.exists():
            audio_path.unlink()

    log("\nDiarization complete!")


if __name__ == "__main__":
    main()
