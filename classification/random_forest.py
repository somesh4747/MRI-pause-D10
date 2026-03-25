import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(training_csv, model_output_path='ad_mci_model.pkl'):
    """
    Train a Random Forest classifier for AD/MCI detection
    
    Args:
        training_csv: Path to training dataset CSV
        model_output_path: Path to save trained model
    """
    
    print("="*80)
    print("TRAINING AD/MCI DETECTION MODEL")
    print("="*80)
    
    # Load training data
    print(f"\nLoading training data from: {training_csv}")
    df = pd.read_csv(training_csv)
    
    print(f"Total patients: {len(df)}")
    print(f"\nDiagnosis distribution:")
    print(df['diagnosis_name'].value_counts())
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'diagnosis', 'diagnosis_name']]
    X = df[feature_cols].astype(float)
    y = df['diagnosis'].astype(int)
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} patients")
    print(f"  Test set: {len(X_test)} patients")
    
    # Scale features (important for comparing pause duration with word count, etc.)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    
    print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy:     {accuracy_score(y_test, y_test_pred):.4f}")
    
    print(f"\nTest Set Detailed Metrics:")
    print(f"  Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"\nCross-Validation (5-fold):")
    print(f"  Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    diagnosis_names = {0: 'Control', 1: 'MCI', 2: 'AD'}
    print(classification_report(y_test, y_test_pred, target_names=[diagnosis_names[i] for i in range(3)]))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model and scaler
    model_dir = 'model_artifacts'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'ad_mci_model.pkl')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_names.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    
    print(f"\n{'='*80}")
    print("MODEL SAVED")
    print(f"{'='*80}")
    print(f"Model:        {model_path}")
    print(f"Scaler:       {scaler_path}")
    print(f"Features:     {features_path}")
    
    return model, scaler, feature_cols


# Example usage
if __name__ == '__main__':
    import os
    
    training_csv = r"E:\ML\silero-python\training_dataset.csv"
    train_model(training_csv)