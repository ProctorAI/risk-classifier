import pandas as pd
import numpy as np
from ml.utils.database import fetch_exam_events
from features.mouse_features.extractor import MouseFeatureExtractor
from features.keyboard_features.extractor import KeyboardFeatureExtractor
from features.window_features.extractor import WindowStateFeatureExtractor
from features.scoring import calculate_total_score, get_risk_level
from datetime import timedelta
import os
import pytz

# Create necessary directories
os.makedirs('ml/data', exist_ok=True)

def verify_risk_scores(df: pd.DataFrame) -> None:
    """Verify that risk scores are properly calculated and present"""
    expected_columns = [
        'mouse_score', 'keyboard_score', 'window_score',
        'total_risk_score', 'risk_level'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"\nWARNING: Missing risk score columns: {missing_cols}")
        return False
    
    print("\nRisk Score Verification:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with scores: {df['total_risk_score'].notna().sum()}")
    print("\nScore Ranges:")
    for col in ['mouse_score', 'keyboard_score', 'window_score', 'total_risk_score']:
        if col in df.columns:
            print(f"{col}: {df[col].min():.2f} - {df[col].max():.2f}")
    
    if 'risk_level' in df.columns:
        print("\nRisk Level Distribution:")
        print(df['risk_level'].value_counts())
    
    return True

def process_exam_data(events: list, window_size: int = 30) -> pd.DataFrame:
    """Process exam events and extract features
    
    Args:
        events: List of events from database
        window_size: Size of time window in seconds
        
    Returns:
        DataFrame with extracted features and risk scores
    """
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    # Debug print
    print("\nData structure example:")
    print("Columns:", df.columns.tolist())
    print("\nFirst row data column type:", type(df['data'].iloc[0]))
    print("First row data content:", df['data'].iloc[0])
    
    # Convert timestamps to IST
    ist = pytz.timezone('Asia/Kolkata')
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(ist)
    
    # Initialize feature extractors
    mouse_extractor = MouseFeatureExtractor(window_size=window_size)
    keyboard_extractor = KeyboardFeatureExtractor(window_size=window_size)
    window_extractor = WindowStateFeatureExtractor(window_size=window_size)
    feature_windows = []
    
    # Process each exam separately
    for exam_id in df['exam_id'].unique():
        if not exam_id:  # Skip if exam_id is None
            continue
            
        print(f"Processing exam: {exam_id}")
        exam_data = df[df['exam_id'] == exam_id].copy()
        
        # Process each time window
        start_time = exam_data['created_at'].min()
        end_time = exam_data['created_at'].max()
        
        current_window_start = start_time
        while current_window_start < end_time:
            current_window_end = current_window_start + timedelta(seconds=window_size)
            
            # Get window data
            window_data = exam_data[
                (exam_data['created_at'] >= current_window_start) & 
                (exam_data['created_at'] < current_window_end)
            ]
            
            if not window_data.empty:
                # Extract features
                mouse_features = mouse_extractor.extract_features(window_data)
                keyboard_features = keyboard_extractor.extract_features(window_data)
                window_features = window_extractor.extract_features(window_data)
                
                # Combine features
                features = {
                    **mouse_features,
                    **keyboard_features,
                    **window_features,
                    'exam_id': exam_id,
                    'window_start': current_window_start
                }
                
                # Calculate risk scores
                total_score, category_scores = calculate_total_score(features)
                features.update(category_scores)
                features['total_risk_score'] = total_score
                features['risk_level'] = get_risk_level(total_score)
                
                feature_windows.append(features)
            
            current_window_start = current_window_end
    
    # Convert to DataFrame and ensure score columns exist
    features_df = pd.DataFrame(feature_windows)
    score_columns = ['mouse_score', 'keyboard_score', 'window_score', 
                    'total_risk_score', 'risk_level']
    for col in score_columns:
        if col not in features_df.columns:
            print(f"\nWARNING: Missing column {col}")
    
    return features_df

def main():
    # Fetch events from database
    print("Fetching events from database...")
    events = fetch_exam_events()
    
    if not events:
        print("No events found in database")
        return
        
    print(f"Found {len(events)} events")
    
    # Process events and extract features
    print("Extracting features...")
    features_df = process_exam_data(events)
    
    # Verify risk scores before saving
    if not verify_risk_scores(features_df):
        print("\nWARNING: Risk scores may not be properly calculated!")
    
    # Save features
    output_file = 'ml/data/extracted_features-1.csv'
    
    # Ensure all score columns are present
    score_columns = ['mouse_score', 'keyboard_score', 'window_score', 
                    'total_risk_score', 'risk_level']
    for col in score_columns:
        if col not in features_df.columns:
            features_df[col] = np.nan
    
    # Save with scores first for easy viewing
    column_order = (
        score_columns + 
        [col for col in features_df.columns if col not in score_columns]
    )
    features_df[column_order].to_csv(output_file, index=False)
    
    print(f"\nFeatures saved to {output_file}")
    print(f"Total windows processed: {len(features_df)}")
    
    # Print feature statistics
    print("\nFeature statistics:")
    stats = features_df.describe().round(4)
    
    # Risk score summary first
    print("\nRisk Score Summary:")
    risk_cols = ['mouse_score', 'keyboard_score', 'window_score', 'total_risk_score']
    print(stats[risk_cols])
    
    # Risk level distribution
    risk_dist = features_df['risk_level'].value_counts()
    print("\nRisk Level Distribution:")
    print(risk_dist)
    
    # Mouse feature statistics
    print("\nMouse Features:")
    mouse_cols = ['avg_norm_x', 'avg_norm_y', 'std_norm_x', 'std_norm_y',
                 'top_edge_time', 'bottom_edge_time', 'idle_percentage', 'mouse_score']
    print(stats[mouse_cols])
    
    # Keyboard feature statistics
    print("\nKeyboard Features:")
    keyboard_cols = ['key_press_rate', 'shortcut_key_ratio', 'backspace_ratio',
                    'rapid_key_ratio', 'clipboard_operation_rate', 'keyboard_score']
    print(stats[keyboard_cols])
    
    # Window state statistics
    print("\nWindow State Features:")
    window_cols = ['blur_count', 'tab_switch_count', 'window_resize_count',
                  'rapid_switch_count', 'total_blur_duration', 'window_score']
    print(stats[window_cols])
    
    # Print exam summary with focus on risk scores
    print("\nExams processed:")
    exam_summary = features_df.groupby('exam_id').agg({
        'window_start': 'count',
        'total_risk_score': ['mean', 'max', 'min'],
        'mouse_score': ['mean', 'max'],
        'keyboard_score': ['mean', 'max'],
        'window_score': ['mean', 'max']
    }).round(4)
    
    # Flatten column names
    exam_summary.columns = [
        'num_windows',
        'avg_risk', 'max_risk', 'min_risk',
        'avg_mouse', 'max_mouse',
        'avg_keyboard', 'max_keyboard',
        'avg_window', 'max_window'
    ]
    print(exam_summary)

if __name__ == "__main__":
    main() 