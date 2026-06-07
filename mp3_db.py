"""

MP3 DB 🎵

Recursive MP3 metadata and audio analysis utility.

Built by Jaime Chica
https://github.com/jc-labx

"""

import os
import time
import traceback
import pandas as pd
import pycountry

from pathlib import Path
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, ID3NoHeaderError

from pydub import AudioSegment

# librosa can be slower/heavier; used for BPM estimation
import numpy as np
import librosa

# -----------------------------
# Configuration (tweak if needed)
# -----------------------------
SILENCE_THRESH_DBFS = -45.0   # silence threshold in dBFS (more negative => stricter)
SILENCE_CHUNK_MS = 10         # analysis granularity in milliseconds
BPM_ANALYZE_MAX_SECONDS = 120  # analyze up to first N seconds for BPM (speed optimization)

ALLOWED_TAGS = {"TIT2", "TPE1", "TALB", "TYER", "TDRC", "TCON", "COMM", "TCOM", "APIC"}

# -----------------------------
# Helper functions
# -----------------------------
def safe_str(x):
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def is_country_name(text: str) -> bool:
    """
    Return True if text corresponds to a country name/alias/code recognized by pycountry.
    """
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    try:
        # pycountry lookup handles names, alpha_2, alpha_3, numeric in many cases
        pycountry.countries.lookup(t)
        return True
    except Exception:
        return False


def frame_to_readable_value(frame) -> str:
    """
    Convert a mutagen ID3 frame object into a readable string.
    """
    # Text frames typically have .text
    if hasattr(frame, "text"):
        try:
            return "; ".join([safe_str(v) for v in frame.text])
        except Exception:
            pass

    # URL frames may have .url
    if hasattr(frame, "url"):
        try:
            return safe_str(frame.url)
        except Exception:
            pass

    # Comments (COMM) have .text too, but also .lang/.desc
    # If not handled above, fall back:
    try:
        return safe_str(frame)
    except Exception:
        return repr(frame)


def get_first_text_frame(tags: ID3, frame_id: str) -> str:
    frames = tags.getall(frame_id) if tags else []
    if not frames:
        return ""
    # Most text frames provide .text as a list
    f = frames[0]
    if hasattr(f, "text"):
        try:
            if f.text:
                return safe_str(f.text[0])
        except Exception:
            return safe_str(f)
    return safe_str(f)


def get_year(tags: ID3) -> str:
    """
    Prefer TDRC (recording time) if present, otherwise TYER.
    """
    y = get_first_text_frame(tags, "TDRC").strip()
    if y:
        # Sometimes it includes full date; keep as-is, or you can slice first 4 chars if desired
        return y
    return get_first_text_frame(tags, "TYER").strip()


def get_comment(tags: ID3) -> str:
    """
    Combine all COMM frames into a single string. If multiple, join with ' | '.
    """
    if not tags:
        return ""
    comms = tags.getall("COMM")
    if not comms:
        return ""
    parts = []
    for c in comms:
        # mutagen COMM typically has .text list; .desc and .lang
        try:
            txt = "; ".join([safe_str(v) for v in getattr(c, "text", [])]) or safe_str(c)
        except Exception:
            txt = safe_str(c)
        if txt.strip():
            parts.append(txt.strip())
    return " | ".join(parts)


def count_cover_images(tags: ID3) -> int:
    if not tags:
        return 0
    return len(tags.getall("APIC"))


def other_tags_and_values(tags: ID3) -> str:
    """
    If the file includes any other ID3 tag different than ALLOWED_TAGS,
    list 'TAGID: value' lines separated by line breaks.
    """
    if not tags:
        return ""
    lines = []
    for frame in tags.values():
        fid = getattr(frame, "FrameID", None)
        if not fid:
            continue
        if fid in ALLOWED_TAGS:
            continue
        val = frame_to_readable_value(frame)
        # Keep it readable; avoid extremely long binary-like outputs
        if val and len(val) > 1000:
            val = val[:1000] + "…"
        lines.append(f"{fid}: {val}")
    return "\n".join(lines)


def detect_leading_silence_seconds(audio: AudioSegment, silence_thresh_dbfs: float, chunk_ms: int) -> float:
    """
    Detect leading silence by scanning chunks until audio is above threshold.
    Returns seconds.
    """
    trim_ms = 0
    # Handle fully silent tracks
    if audio.dBFS == float("-inf"):
        return len(audio) / 1000.0

    while trim_ms < len(audio):
        chunk = audio[trim_ms:trim_ms + chunk_ms]
        if chunk.dBFS > silence_thresh_dbfs:
            break
        trim_ms += chunk_ms
    return trim_ms / 1000.0


def detect_trailing_silence_seconds(audio: AudioSegment, silence_thresh_dbfs: float, chunk_ms: int) -> float:
    """
    Detect trailing silence by scanning from the end backwards.
    Returns seconds.
    """
    trim_ms = 0
    if audio.dBFS == float("-inf"):
        return len(audio) / 1000.0

    # Reverse by slicing from the end
    total = len(audio)
    while trim_ms < total:
        start = max(total - (trim_ms + chunk_ms), 0)
        end = total - trim_ms
        chunk = audio[start:end]
        if chunk.dBFS > silence_thresh_dbfs:
            break
        trim_ms += chunk_ms
    return trim_ms / 1000.0


def estimate_bpm(path: Path, max_seconds: int = BPM_ANALYZE_MAX_SECONDS):
    """
    Estimate BPM using librosa beat tracking.
    For performance, analyze at most the first max_seconds seconds.
    """
    try:
        # librosa loads to float32 mono; uses audioread/ffmpeg backend depending on system
        y, sr = librosa.load(path, mono=True, duration=max_seconds)
        if y is None or len(y) < sr * 5:
            return None  # too short
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # tempo can be numpy scalar
        if tempo is None:
            return None
        tempo_val = float(np.asarray(tempo).squeeze())
        if np.isnan(tempo_val) or tempo_val <= 0:
            return None
        return round(tempo_val, 2)
    except Exception:
        return None


def read_mp3_info(path: Path):
    """
    Read ID3 tags and technical info. Returns dict with required columns.
    """
    # --- File basics
    file_name = path.name
    file_dir = str(path.parent.resolve())

    # --- ID3 tags
    tags = None
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = None
    except Exception:
        tags = None

    song_title = get_first_text_frame(tags, "TIT2").strip() if tags else ""
    artist = get_first_text_frame(tags, "TPE1").strip() if tags else ""
    album = get_first_text_frame(tags, "TALB").strip() if tags else ""
    year = get_year(tags) if tags else ""
    genre = get_first_text_frame(tags, "TCON").strip() if tags else ""
    comment = get_comment(tags).strip()

    wrong_country = "yes" if (comment and not is_country_name(comment)) else "no"

    other_tags = other_tags_and_values(tags)
    cover_count = count_cover_images(tags)

    # --- Technical info: bitrate & length from audio stream (not ID3 tags)
    bitrate_kbps = None
    audio_length_sec = None
    try:
        mp3 = MP3(path)
        if mp3.info:
            if getattr(mp3.info, "bitrate", None) is not None:
                bitrate_kbps = int(mp3.info.bitrate / 1000)
            if getattr(mp3.info, "length", None) is not None:
                audio_length_sec = float(mp3.info.length)
    except Exception:
        bitrate_kbps = None
        audio_length_sec = None

    # --- Silence detection (requires decoding)
    leading_silence = None
    trailing_silence = None
    try:
        audio = AudioSegment.from_file(path, format="mp3")
        leading_silence = round(
            detect_leading_silence_seconds(audio, SILENCE_THRESH_DBFS, SILENCE_CHUNK_MS), 3
        )
        trailing_silence = round(
            detect_trailing_silence_seconds(audio, SILENCE_THRESH_DBFS, SILENCE_CHUNK_MS), 3
        )
        # If length wasn't obtained from MP3 info, derive it from decoded audio
        if audio_length_sec is None:
            audio_length_sec = round(len(audio) / 1000.0, 3)
    except Exception:
        leading_silence = None
        trailing_silence = None

    # --- BPM estimation
    #bpm = estimate_bpm(path, BPM_ANALYZE_MAX_SECONDS)
    bpm = ""

    return {
        "File Name": file_name,
        "File Directory": file_dir,
        "Song title": song_title,
        "Artist": artist,
        "Album": album,
        "Year": year,
        "Genre": genre,
        "Comment": comment,
        "Wrong country": wrong_country,
        "Other Tags": other_tags,
        "Album Arts": cover_count,
        "Bit rate": bitrate_kbps,
        "Audio length": (
            "" if audio_length_sec is None
            else f"{int(audio_length_sec)//60:02d}:{int(audio_length_sec)%60:02d}"),
        "Leading silence": leading_silence,
        "Trailing silence": trailing_silence,
        "BPM estimation": bpm,
    }


def main():
    root = Path(".").resolve()

    # Find mp3 files recursively, case-insensitive
    mp3_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3"]

    rows = []
    errors = []

    print(f"Scanning: {root} -> {len(mp3_files)} .mp3 file(s) Found!")

    for i, path in enumerate(mp3_files, start=1):
        try:
            print(str(path) + "...")                        
            row = read_mp3_info(path)
            rows.append(row)
        except Exception as e:
            errors.append({
                "file": str(path),
                "error": safe_str(e),
                "trace": traceback.format_exc()
            })

    print(f"\n{i}/{len(mp3_files)} processed!")

    df = pd.DataFrame(rows)

    # Ensure column order exactly as requested
    columns = [
        "File Name",
        "File Directory",
        "Song title",
        "Artist",
        "Album",
        "Year",
        "Genre",
        "Comment",
        "Wrong country",
        "Other Tags",
        "Album Arts",
        "Bit rate",
        "Audio length",
        "Leading silence",
        "Trailing silence",
        "BPM estimation",
    ]
    for c in columns:
        if c not in df.columns:
            df[c] = None
    df = df[columns]

    # Output filename (no input parameter)
    out_name = "mp3db_" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".xlsx"

    # Write Excel
    with pd.ExcelWriter(out_name, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="mp3db")

        # Optional: store errors in a second sheet for troubleshooting
        if errors:
            err_df = pd.DataFrame(errors)
            err_df.to_excel(writer, index=False, sheet_name="errors")
        

if __name__ == "__main__":
    main()
