import pandas as pd


def get_patient_word_segments(file_path):
    """
    Extract all patient (PAR) words with their individual timings from a .cha file.
    Returns a list of word segments with PAR utterance tracking.
    Also extracts silence information within each PAR line.
    """
    word_segments = []
    word_count = 0
    par_count = 0
    par_data = []  # Track data per PAR utterance

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"Total lines in file: {len(lines)}")

        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n")

            # Look for patient utterance lines
            if line.startswith("*PAR:"):
                par_count += 1
                par_content = line.replace("*PAR:", "").strip()
                # print(
                #     f"\nFound PAR utterance #{par_count} at line {i}: {par_content[:50]}"
                # )

                # Look ahead for the %wor: line (may be after %mor: and %gra: lines)
                j = i + 1
                found_wor = False
                par_words = []

                while j < len(lines) and j < i + 10:  # Look within next 10 lines
                    next_line = lines[j].rstrip("\n")

                    if next_line.startswith("%wor:"):
                        found_wor = True
                        print(f"  Found %wor line!")
                        # Parse the word timings
                        wor_content = next_line.replace("%wor:", "").strip()

                        # Split by spaces to get words and timings
                        parts = wor_content.split()
                        print(f"  Parts count: {len(parts)}")

                        k = 0
                        while k < len(parts):
                            word = parts[k]

                            # Check if next item is a timing (contains underscore)
                            # The timing might have tab characters, so clean it
                            if k + 1 < len(parts):
                                timing_raw = parts[k + 1]
                                # Remove tab characters and other whitespace
                                timing = timing_raw.replace("\x15", "").strip()

                                if "_" in timing:
                                    try:
                                        start_ms, end_ms = map(float, timing.split("_"))

                                        # Convert milliseconds to seconds
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
                                    except ValueError as e:
                                        print(
                                            f"    Error parsing timing '{timing}': {e}"
                                        )
                                        k += 1
                                else:
                                    k += 1
                            else:
                                k += 1

                        # Calculate silences within this PAR line
                        if par_words:
                            par_data.append(
                                {
                                    "par_num": par_count,
                                    "par_text": par_content,
                                    "words": par_words,
                                    "total_duration": par_words[-1]["end_sec"]
                                    - par_words[0]["start_sec"],
                                }
                            )

                        break
                    elif next_line.startswith("*"):
                        # Reached next utterance without finding %wor
                        print(f"  Reached next utterance without %wor")
                        break
                    else:
                        j += 1

                if not found_wor:
                    print(f"  Warning: No %wor line found for this PAR utterance")

            i += 1

        print(f"\n\nTotal PAR utterances found: {par_count}")
        print(f"Total words extracted: {word_count}")

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback

        traceback.print_exc()

    return word_segments, par_data


def print_word_segments(segments):
    """Print word-level segments in a readable format"""

    print("=" * 110)
    print("PATIENT VOICE SEGMENTS - WORD BY WORD (Ready for Chopping)")
    print("=" * 110)
    print(f"\nTotal Patient Words: {len(segments)}\n")

    total_voice_time = 0

    for seg in segments:
        total_voice_time += seg["duration_sec"]
        print(
            f"Word {seg['word_num']:4d}: '{seg['word']:20s}' | "
            f"Start: {seg['start_sec']:8.3f}s | End: {seg['end_sec']:8.3f}s | "
            f"Duration: {seg['duration_sec']:6.3f}s"
        )

    print("\n" + "=" * 110)
    print(
        f"Total Patient Voice Time: {total_voice_time:.2f} seconds ({total_voice_time/60:.2f} minutes)"
    )
    print(
        f"Average Word Duration: {total_voice_time/len(segments):.3f} seconds"
        if segments
        else "No segments"
    )
    print("=" * 110)


def save_word_segments(segments, output_file):
    """Save word segments to CSV for easy use with audio processing tools"""

    df = pd.DataFrame(segments)
    df.to_csv(output_file, index=False)
    print(f"\nWord segments saved to: {output_file}")

    # Also create a script-friendly format for FFmpeg
    # script_file = output_file.replace(".csv", "_ffmpeg_commands.txt")

    # with open(script_file, "w") as f:
    #     f.write("# FFmpeg commands to extract patient voice segments - WORD BY WORD\n")
    #     f.write("# Usage: ffmpeg -i input.mp3 -ss START -to END -c copy output.mp3\n\n")

    #     for seg in segments:
    #         f.write(f"# Word {seg['word_num']}: {seg['word']}\n")
    #         f.write(f"ffmpeg -i input.mp3 -ss {seg['start_sec']} -to {seg['end_sec']} ")
    #         f.write(f"-c copy word_{seg['word_num']:04d}_{seg['word']}.mp3\n\n")

    # print(f"FFmpeg commands saved to: {script_file}")


def get_word_segments_as_list(segments):
    """Return segments as a simple list for programmatic use"""
    return [(seg["word"], seg["start_sec"], seg["end_sec"]) for seg in segments]


def create_silence_map(segments, par_data):
    """
    Create a map of silence (gaps) WITHIN each PAR utterance.
    Only counts silences between words in the same PAR line, not between PAR lines.

    Returns:
        - silences: List of silence gaps within each PAR line
        - par_silence_summary: Summary of total silence per PAR utterance
    """
    if not par_data:
        return [], []

    silences = []
    par_silence_summary = []

    for par in par_data:
        par_num = par["par_num"]
        words = par["words"]
        par_total_silence = 0

        # Calculate silences between consecutive words WITHIN this PAR line
        for i in range(len(words) - 1):
            current_end = words[i]["end_sec"]
            next_start = words[i + 1]["start_sec"]

            silence_duration = next_start - current_end

            if silence_duration > 0:  # Only if there's a gap
                silences.append(
                    {
                        "par_num": par_num,
                        "par_text": par["par_text"][:50],  # First 50 chars
                        "between_word": f"{words[i]['word']} -> {words[i + 1]['word']}",
                        "silence_start": round(current_end, 3),
                        "silence_end": round(next_start, 3),
                        "silence_duration_sec": round(silence_duration, 3)
                    }
                )
                par_total_silence += silence_duration

        # Calculate PAR-level statistics
        par_total_duration = par["total_duration"]
        par_speech_duration = par_total_duration - par_total_silence

        par_silence_summary.append(
            {
                "par_num": par_num,
                "par_text": par["par_text"][:100],
                "total_duration_sec": round(par_total_duration, 3),
                "total_silence_sec": round(par_total_silence, 3),
                "total_speech_sec": round(par_speech_duration, 3),
                "silence_percentage": round(
                    (
                        (par_total_silence / par_total_duration * 100)
                        if par_total_duration > 0
                        else 0
                    ),
                    2,
                ),
                "num_words": len(words),
                "num_silences": len(words) - 1,  # n-1 gaps for n words
            }
        )

    return silences, par_silence_summary


# Example usage


def get_response_time(file_path):
    """
    Extract response time between INV (investigator) and PAR (patient) utterances.
    Only calculate response time when a PAR's IMMEDIATELY PRECEDING utterance is an INV.
    If consecutive PARs follow, only the first PAR (right after INV) gets a response time.
    If consecutive INVs appear, only the LAST INV (right before PAR) is used.
    Response time = PAR start time - INV end time
    
    Returns:
        - response_times: List of response time data
        - inv_par_pairs: Raw data of INV-PAR pairs (ONE pair per INV)
    """
    response_times = []
    inv_par_pairs = []
    
    utterances_list = []  # Store all utterances in file order
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        i = 0
        inv_count = 0
        par_count = 0
        
        while i < len(lines):
            line = lines[i].rstrip("\n")
            
            # ===== EXTRACT INV UTTERANCES =====
            if line.startswith("*INV:"):
                inv_count += 1
                inv_content = line.replace("*INV:", "").strip()
                
                # Look for %wor: line
                j = i + 1
                inv_words = []
                
                while j < len(lines) and j < i + 10:
                    next_line = lines[j].rstrip("\n")
                    
                    if next_line.startswith("%wor:"):
                        wor_content = next_line.replace("%wor:", "").strip()
                        parts = wor_content.split()
                        
                        # Parse words and timings
                        k = 0
                        while k < len(parts):
                            word = parts[k]
                            
                            if k + 1 < len(parts):
                                timing_raw = parts[k + 1]
                                timing = timing_raw.replace("\x15", "").strip()
                                
                                if "_" in timing:
                                    try:
                                        start_ms, end_ms = map(float, timing.split("_"))
                                        inv_words.append({
                                            "word": word,
                                            "start_ms": start_ms,
                                            "end_ms": end_ms,
                                            "start_sec": start_ms / 1000.0,
                                            "end_sec": end_ms / 1000.0
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
                        "words": inv_words,
                        "start_sec": inv_words[0]["start_sec"],
                        "end_sec": inv_words[-1]["end_sec"]
                    })
            
            # ===== EXTRACT PAR UTTERANCES =====
            elif line.startswith("*PAR:"):
                par_count += 1
                par_content = line.replace("*PAR:", "").strip()
                
                # Look for %wor: line
                j = i + 1
                par_words = []
                
                while j < len(lines) and j < i + 10:
                    next_line = lines[j].rstrip("\n")
                    
                    if next_line.startswith("%wor:"):
                        wor_content = next_line.replace("%wor:", "").strip()
                        parts = wor_content.split()
                        
                        # Parse words and timings
                        k = 0
                        while k < len(parts):
                            word = parts[k]
                            
                            if k + 1 < len(parts):
                                timing_raw = parts[k + 1]
                                timing = timing_raw.replace("\x15", "").strip()
                                
                                if "_" in timing:
                                    try:
                                        start_ms, end_ms = map(float, timing.split("_"))
                                        par_words.append({
                                            "word": word,
                                            "start_ms": start_ms,
                                            "end_ms": end_ms,
                                            "start_sec": start_ms / 1000.0,
                                            "end_sec": end_ms / 1000.0
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
                        "words": par_words,
                        "start_sec": par_words[0]["start_sec"],
                        "end_sec": par_words[-1]["end_sec"]
                    })
            
            i += 1
        
        # Only create a response time when a PAR's IMMEDIATELY PRECEDING utterance is an INV.
        # If multiple consecutive INVs appear before a PAR, use the LAST INV (the one right before PAR).
        # If a PAR follows another PAR, skip it - no response time entry.
        for i, utterance in enumerate(utterances_list):
            if utterance["type"] == "PAR" and i > 0:
                # Check if the IMMEDIATELY preceding utterance is an INV
                prev = utterances_list[i - 1]
                if prev["type"] == "INV":
                    response_time_sec = utterance["start_sec"] - prev["end_sec"]
                    
                    inv_par_pairs.append({
                        "inv_num": prev["inv_num"],
                        "par_num": utterance["par_num"],
                        "inv_text": prev["inv_text"][:50],
                        "par_text": utterance["par_text"][:50],
                        "inv_end_sec": round(prev["end_sec"], 3),
                        "par_start_sec": round(utterance["start_sec"], 3),
                        "response_time_sec": round(response_time_sec, 3),
                        "inv_duration_sec": round(prev["end_sec"] - prev["start_sec"], 3),
                        "par_duration_sec": round(utterance["end_sec"] - utterance["start_sec"], 3)
                    })
                    
                    response_times.append({
                        "inv_num": prev["inv_num"],
                        "par_num": utterance["par_num"],
                        "response_time_sec": round(response_time_sec, 3)
                    })
        
        return response_times, inv_par_pairs
        
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def save_response_time_data(inv_par_pairs, output_file):
    """Save response time data to CSV"""
    if not inv_par_pairs:
        print("No response time data to save!")
        return
    
    df = pd.DataFrame(inv_par_pairs)
    df.to_csv(output_file, index=False)
    print(f"\nResponse time data saved to: {output_file}")
    
    # Print summary statistics
    if "response_time_sec" in df.columns:
        print(f"\n{'='*100}")
        print("RESPONSE TIME SUMMARY STATISTICS")
        print(f"{'='*100}")
        print(f"Total INV-PAR pairs: {len(df)}")
        print(f"Mean response time: {df['response_time_sec'].mean():.3f} seconds")
        print(f"Median response time: {df['response_time_sec'].median():.3f} seconds")
        print(f"Min response time: {df['response_time_sec'].min():.3f} seconds")
        print(f"Max response time: {df['response_time_sec'].max():.3f} seconds")
        print(f"Std dev: {df['response_time_sec'].std():.3f} seconds")
        print(f"{'='*100}\n")
        
        # Show top 10 longest response times
        print("Top 10 Longest Response Times:")
        print(df.nlargest(10, 'response_time_sec')[['inv_num', 'par_num', 'response_time_sec']].to_string(index=False))


def get_report(file_path):
    # file_path = r"E:\ML\silero-python\Delaware\MCI\01-1.cha"

    # Get all patient word segments and PAR data
    word_segments, par_data = get_patient_word_segments(file_path)

    if word_segments:
        # Print analysis
        # print_word_segments(word_segments)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Save to CSV and FFmpeg commands
        output_csv = r"E:\ML\silero-python\patient_word_segments.csv"
        # save_word_segments(word_segments, output_csv)
        # =============================================================

        # Get as simple list for programmatic use
        word_list = get_word_segments_as_list(word_segments)
        

        # Create silence map (WITHIN each PAR line only)
        silences, par_silence_summary = create_silence_map(word_segments, par_data)

        if par_silence_summary:
            # print(f"\n\n{'='*120}")
            # print(
            #     "SILENCE ANALYSIS - BY PAR UTTERANCE (Only silences WITHIN each PAR line)"
            # )
            # print(f"{'='*120}")

            # Save PAR-level silence summary
            silence_summary_df = pd.DataFrame(par_silence_summary)
            silence_summary_csv = output_csv.replace(".csv", "_par_silence_summary.csv")
            silence_summary_df.to_csv(silence_summary_csv, index=False)
            # print(f"\nPAR-level silence summary saved to: {silence_summary_csv}")

            # print(f"\nTop 10 PAR Utterances by Total Silence:")
            # print(
            #     silence_summary_df.sort_values("total_silence_sec", ascending=False)
            #     .head(10)
            #     .to_string(index=False)
            # )

            total_overall_silence = silence_summary_df["total_silence_sec"].sum()
            total_overall_speech = silence_summary_df["total_speech_sec"].sum()
            total_overall_duration = silence_summary_df["total_duration_sec"].sum()
            total_par_count = len(par_silence_summary)

            

            # Save detailed silence map
            if silences:
                silence_df = pd.DataFrame(silences)
                silence_csv = output_csv.replace(".csv", "_silences_detailed.csv")
                silence_df.to_csv(silence_csv, index=False)
                # print(f"\nDetailed silence map saved to: {silence_csv}")

                # print(f"\nTop 10 Longest Individual Silences:")
                # silence_df_sorted = silence_df.sort_values(
                #     "silence_duration_sec", ascending=False
                # )
                # print(
                #     silence_df_sorted.head(10)[
                #         ["par_num", "between_word", "silence_duration_sec"]
                #     ].to_string(index=False)
                # )
            
            # ====== Calculate response time between INV and PAR =======
            response_times, inv_par_pairs = get_response_time(file_path)
            
            if inv_par_pairs:
                response_time_csv = output_csv.replace(".csv", "_response_time.csv")
                # save_response_time_data(response_times, response_time_csv)
            
            # ====== returning the silence summary =======
            return silences, par_silence_summary, word_segments, response_times
        else:
            print("No PAR utterances with silence found!")
            return [], [], [], []
    else:
        print("No patient word segments found!")
        return [], [], [], []


if __name__ == "__main__":
    get_report(r"E:\ML\silero-python\Delaware\MCI\01-1.cha")
