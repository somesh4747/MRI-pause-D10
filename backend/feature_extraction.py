"""
Feature extraction from .cha files.

Self-contained module — bundles the parsing logic from pause_cha_word_by_word.py
and _main_features.py so the backend has no external dependencies on the parent
project files.
"""


# ──────────────────────────────────────────────────────────────────────────────
# .cha file parsing (from pause_cha_word_by_word.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_patient_word_segments(file_path: str):
    """
    Extract all patient (PAR) words with their individual timings from a .cha file.
    Also extracts silence information within each PAR line.
    Accepts a file path (str) or file-like content lines.
    """
    word_segments = []
    word_count = 0
    par_count = 0
    par_data = []

    # Support both file path and pre-read lines
    if isinstance(file_path, str):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = file_path  # already a list of lines

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        if line.startswith("*PAR:"):
            par_count += 1
            par_content = line.replace("*PAR:", "").strip()

            j = i + 1
            found_wor = False
            par_words = []

            while j < len(lines) and j < i + 10:
                next_line = lines[j].rstrip("\n")

                if next_line.startswith("%wor:"):
                    found_wor = True
                    wor_content = next_line.replace("%wor:", "").strip()
                    parts = wor_content.split()

                    k = 0
                    while k < len(parts):
                        word = parts[k]

                        if k + 1 < len(parts):
                            timing_raw = parts[k + 1]
                            timing = timing_raw.replace("\x15", "").strip()

                            if "_" in timing:
                                try:
                                    start_ms, end_ms = map(float, timing.split("_"))
                                    start_sec = start_ms / 1000.0
                                    end_sec = end_ms / 1000.0
                                    duration_sec = end_sec - start_sec

                                    word_count += 1

                                    word_segment = {
                                        "word_num": word_count,
                                        "word": word,
                                        "start_ms": start_ms,
                                        "end_ms": end_ms,
                                        "start_sec": round(start_sec, 3),
                                        "end_sec": round(end_sec, 3),
                                        "duration_sec": round(duration_sec, 3),
                                        "par_num": par_count,
                                    }
                                    word_segments.append(word_segment)
                                    par_words.append(word_segment)
                                    k += 2
                                except ValueError:
                                    k += 1
                            else:
                                k += 1
                        else:
                            k += 1

                    if par_words:
                        par_data.append({
                            "par_num": par_count,
                            "par_text": par_content,
                            "words": par_words,
                            "total_duration": par_words[-1]["end_sec"] - par_words[0]["start_sec"],
                        })
                    break
                elif next_line.startswith("*"):
                    break
                else:
                    j += 1

            i += 1
            continue
        i += 1

    return word_segments, par_data


def create_silence_map(segments, par_data):
    """
    Create a map of silence (gaps) WITHIN each PAR utterance.
    Only counts silences between words in the same PAR line, not between PAR lines.
    """
    if not par_data:
        return [], []

    silences = []
    par_silence_summary = []

    for par in par_data:
        par_num = par["par_num"]
        words = par["words"]
        par_total_silence = 0

        for i in range(len(words) - 1):
            current_end = words[i]["end_sec"]
            next_start = words[i + 1]["start_sec"]
            silence_duration = next_start - current_end

            if silence_duration > 0:
                silences.append({
                    "par_num": par_num,
                    "par_text": par["par_text"][:50],
                    "between_word": f"{words[i]['word']} -> {words[i + 1]['word']}",
                    "silence_start": round(current_end, 3),
                    "silence_end": round(next_start, 3),
                    "silence_duration_sec": round(silence_duration, 3),
                })
                par_total_silence += silence_duration

        par_total_duration = par["total_duration"]
        par_speech_duration = par_total_duration - par_total_silence

        par_silence_summary.append({
            "par_num": par_num,
            "par_text": par["par_text"][:100],
            "total_duration_sec": round(par_total_duration, 3),
            "total_silence_sec": round(par_total_silence, 3),
            "total_speech_sec": round(par_speech_duration, 3),
            "silence_percentage": round(
                (par_total_silence / par_total_duration * 100) if par_total_duration > 0 else 0,
                2,
            ),
            "num_words": len(words),
            "num_silences": len(words) - 1,
        })

    return silences, par_silence_summary


def get_response_time(lines):
    """
    Extract response time between INV and PAR utterances.
    Accepts a list of lines (already read from file).
    """
    response_times = []
    utterances_list = []

    i = 0
    inv_count = 0
    par_count = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # ── INV utterances ──
        if line.startswith("*INV:"):
            inv_count += 1
            inv_content = line.replace("*INV:", "").strip()
            j = i + 1
            inv_words = []

            while j < len(lines) and j < i + 10:
                next_line = lines[j].rstrip("\n")
                if next_line.startswith("%wor:"):
                    wor_content = next_line.replace("%wor:", "").strip()
                    parts = wor_content.split()
                    k = 0
                    while k < len(parts):
                        word = parts[k]
                        if k + 1 < len(parts):
                            timing = parts[k + 1].replace("\x15", "").strip()
                            if "_" in timing:
                                try:
                                    start_ms, end_ms = map(float, timing.split("_"))
                                    inv_words.append({
                                        "start_sec": start_ms / 1000.0,
                                        "end_sec": end_ms / 1000.0,
                                    })
                                    k += 2
                                except ValueError:
                                    k += 1
                            else:
                                k += 1
                        else:
                            k += 1
                    break
                elif next_line.startswith("*"):
                    break
                else:
                    j += 1

            if inv_words:
                utterances_list.append({
                    "type": "INV",
                    "inv_num": inv_count,
                    "inv_text": inv_content,
                    "start_sec": inv_words[0]["start_sec"],
                    "end_sec": inv_words[-1]["end_sec"],
                })

        # ── PAR utterances ──
        elif line.startswith("*PAR:"):
            par_count += 1
            par_content = line.replace("*PAR:", "").strip()
            j = i + 1
            par_words = []

            while j < len(lines) and j < i + 10:
                next_line = lines[j].rstrip("\n")
                if next_line.startswith("%wor:"):
                    wor_content = next_line.replace("%wor:", "").strip()
                    parts = wor_content.split()
                    k = 0
                    while k < len(parts):
                        word = parts[k]
                        if k + 1 < len(parts):
                            timing = parts[k + 1].replace("\x15", "").strip()
                            if "_" in timing:
                                try:
                                    start_ms, end_ms = map(float, timing.split("_"))
                                    par_words.append({
                                        "start_sec": start_ms / 1000.0,
                                        "end_sec": end_ms / 1000.0,
                                    })
                                    k += 2
                                except ValueError:
                                    k += 1
                            else:
                                k += 1
                        else:
                            k += 1
                    break
                elif next_line.startswith("*"):
                    break
                else:
                    j += 1

            if par_words:
                utterances_list.append({
                    "type": "PAR",
                    "par_num": par_count,
                    "par_text": par_content,
                    "start_sec": par_words[0]["start_sec"],
                    "end_sec": par_words[-1]["end_sec"],
                })

        i += 1

    # Calculate response times (PAR immediately after INV)
    for idx, utterance in enumerate(utterances_list):
        if utterance["type"] == "PAR" and idx > 0:
            prev = utterances_list[idx - 1]
            if prev["type"] == "INV":
                rt = utterance["start_sec"] - prev["end_sec"]
                response_times.append({
                    "inv_num": prev["inv_num"],
                    "par_num": utterance["par_num"],
                    "response_time_sec": round(rt, 3),
                })

    return response_times


def get_report_from_lines(lines):
    """
    Run the full .cha analysis pipeline on pre-read file lines.
    Returns: (silences, par_silence_summary, word_segments, response_times)
    """
    word_segments, par_data = get_patient_word_segments(lines)

    if not word_segments:
        return [], [], [], []

    silences, par_silence_summary = create_silence_map(word_segments, par_data)

    if not par_silence_summary:
        return [], [], word_segments, []

    response_times = get_response_time(lines)
    return silences, par_silence_summary, word_segments, response_times


# ──────────────────────────────────────────────────────────────────────────────
# Feature computation (from _main_features.py)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features_from_lines(lines):
    """
    Extract the 6 classification features from already-read .cha file lines.

    Returns: dict with feature values, or None if extraction fails.
    """
    silences, silence_summary, word_segments, res_times = get_report_from_lines(lines)

    if not silences:
        return None

    silences_durations = [s["silence_duration_sec"] for s in silences]

    total_duration = [i["total_duration_sec"] for i in silence_summary]
    total_speech_times = [i["total_speech_sec"] for i in silence_summary]
    total_pause_times = [i["total_silence_sec"] for i in silence_summary]
    no_of_silences = [w["num_silences"] for w in silence_summary]

    features = {
        "pause_count": sum(no_of_silences),
        "total_speech_time": round(sum(total_duration), 4),
        "total_pause_time": round(sum(total_pause_times), 4),
        "mean_word_duration": (
            round(sum(total_speech_times) / len(word_segments), 4) if word_segments else 0
        ),
        "speech_rate_wpm": (
            round((len(word_segments) / sum(total_speech_times)) * 60, 2)
            if sum(total_speech_times) > 0
            else 0
        ),
        "pause_per_word_ratio": (
            round(len(silences) / len(word_segments), 4) if word_segments else 0
        ),
    }

    return features


def extract_features_from_file(file_path: str):
    """Convenience wrapper — reads a .cha file from disk and extracts features."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return extract_features_from_lines(lines)
