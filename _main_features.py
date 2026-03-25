import pandas as pd
import os
import glob
from pause_cha_word_by_word import get_report
from _to_get_total_speech_length import get_patient_voice_segments

def extract_features_from_patient(cha_file):
    """
    Extract pause and speech timing features from a single patient's .cha file
    
    Returns: Dictionary with all computed features
    """
    silences, get_silence_summary, word_segments, res_times = get_report(cha_file)
    # Get word segments
    # word_segments = get_report(cha_file)
    # voice_segments = get_patient_voice_segments(cha_file)
    # VS = [seg['duration_sec'] for seg in voice_segments]

    if not silences:
        print(f"Warning: No word segments found in {cha_file}")
        return None
    

    if not silences:
        silences_durations = [0]
    else:
        silences_durations = [s['silence_duration_sec'] for s in silences]
    
    # Extract word durations
    # word_durations = [w['duration_sec'] for w in word_segments]

    total_duration = [i['total_duration_sec'] for i in get_silence_summary]
    total_speech_times = [i['total_speech_sec'] for i in get_silence_summary]
    total_pause_times = [i['total_silence_sec'] for i in get_silence_summary]
    # word_segments = [w['word_segment'] for w in silences]
    no_of_silences = [w['num_silences'] for w in get_silence_summary]
    res_time = [i['response_time_sec'] for i in res_times]
    # Calculate features - OPTIMIZED (removed weak/redundant features)
    # Analysis shows only these features effectively discriminate Control vs MCI:
    
    features = {
        # Key pause patterns (Cohen's d = 0.572)
        'pause_count': sum(no_of_silences),
        
        # Speech timing (Cohen's d = 0.513)
        'total_speech_time': round(sum(total_duration), 4),
        
        # Pause timing (Cohen's d = 0.316)
        'total_pause_time': round(sum(total_pause_times), 4),

        # response time is added +++++++++++
        # 'avg_res_time' : round(sum(res_time) / len(res_time), 4),
        
        # Speech rate components (Cohen's d = 0.310)
        'mean_word_duration': round(sum(total_speech_times) / len(word_segments), 4) if word_segments else 0,
        
        # Speech rate metric (Cohen's d = 0.304)
        'speech_rate_wpm': round((len(word_segments) / sum(total_speech_times)) * 60, 2) if sum(total_speech_times) > 0 else 0,
        
        # Pause frequency ratio (Cohen's d = 0.289)
        'pause_per_word_ratio': round(len(silences) / len(word_segments), 4) if word_segments else 0,
    }
    
    # REMOVED FEATURES (too weak, not discriminative):
    # ❌ word_count (redundant with pause_count, r=0.999)
    # ❌ median_pause_duration (d=0.032 - useless)
    # ❌ std_word_duration (d=0.088 - too weak)
    # ❌ min_pause_duration (d=0.102 - too weak)
    # ❌ max_pause_duration (d=0.140 - too weak)
    # ❌ mean_pause_duration (d=0.205 - too weak)
    # ❌ pause_variability (d=0.209 - too weak)
    # ❌ std_pause_duration (d=0.235 - too weak)
    
    return features


def create_training_dataset(patients_dir, output_csv, label_file):
    """
    Create training dataset from multiple patients
    
    Args:
        patients_dir: Directory containing patient .cha files
        output_csv: Path to save the training CSV
        label_file: CSV file with columns: patient_id, diagnosis (0=Control, 1=MCI, 2=AD)
    """
    
    # Load labels
    print(f"Loading patient labels from: {label_file}")
    labels_df = pd.read_csv(label_file)
    label_dict = dict(zip(labels_df['patient_id'], labels_df['diagnosis']))
    
    print(f"Loaded {len(label_dict)} patient labels")
    print(f"Diagnoses: {set(labels_df['diagnosis'])}")
    
    # Find all .cha files recursively
    cha_files = glob.glob(os.path.join(patients_dir, "**/*.cha"), recursive=True)
    print(f"\nFound {len(cha_files)} .cha files")
    print(cha_files)
    
    all_features = []
    
    for i, cha_file in enumerate(cha_files):
        patient_id = os.path.basename(cha_file).replace('.cha', '')
        
        print(f"\n[{i+1}/{len(cha_files)}] Processing: {patient_id}")
        print(f"  Path: {cha_file}")
        
        # Get diagnosis for this patient
        if patient_id not in label_dict:
            print(f"  Warning: No diagnosis found for {patient_id}, skipping...")
            continue
        
        diagnosis = label_dict[patient_id]
        
        # Extract features
        features = extract_features_from_patient(cha_file)
        
        if features is not None:
            features['patient_id'] = patient_id
            features['diagnosis'] = diagnosis
            diagnosis_name = {0: 'Control', 1: 'MCI', 2: 'AD'}
            features['diagnosis_name'] = diagnosis_name.get(diagnosis, 'Unknown')
            all_features.append(features)
            print(f"  ✓ Features extracted ({diagnosis_name.get(diagnosis, 'Unknown')})")
        else:
            print(f"  ✗ Failed to extract features")
    
    # Create DataFrame and save
    df_training = pd.DataFrame(all_features)
    df_training.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"Training dataset saved to: {output_csv}")
    print(f"Total patients: {len(df_training)}")
    print(f"\nDiagnosis distribution:")
    print(df_training['diagnosis_name'].value_counts())
    print(f"{'='*80}")
    
    return df_training


# Example usage
if __name__ == '__main__':
    # Directory with patient .cha files
    patients_dir = r"E:\ML\silero-python\Delaware\MCI\_soumodip"
    
    # CSV file with patient diagnoses (you need to create this)
    # Format: patient_id, diagnosis (0=Control, 1=MCI, 2=AD)
    label_file = r"E:\ML\silero-python\chuci.csv"
    
    # Output training CSV
    output_csv = r"E:\ML\silero-python\training_MCI.csv"
    
    # Create training dataset
    df = create_training_dataset(patients_dir, output_csv, label_file)