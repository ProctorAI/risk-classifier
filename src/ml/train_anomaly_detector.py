import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import joblib
from features.mouse_features.extractor import MouseFeatureExtractor
from datetime import timedelta

class MouseBehaviorAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% of behaviors might be suspicious
            random_state=42
        )
        self.dbscan = DBSCAN(
            eps=0.5,  # Distance threshold for clustering
            min_samples=5  # Minimum samples in a cluster
        )
        
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        # Select numerical features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['exam_id', 'window_start']]
        
        # Scale features
        X = self.scaler.fit_transform(features_df[feature_cols])
        return X
    
    def fit(self, features_df: pd.DataFrame):
        """Train the anomaly detection models"""
        X = self.prepare_features(features_df)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X)
        
        # Train DBSCAN
        self.dbscan.fit(X)
        
        return self
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomaly scores for new data"""
        X = self.prepare_features(features_df)
        
        # Get predictions from both models
        if_scores = self.isolation_forest.score_samples(X)
        dbscan_labels = self.dbscan.fit_predict(X)
        
        # Combine scores
        results = features_df.copy()
        results['isolation_forest_score'] = if_scores
        results['is_outlier_if'] = if_scores < np.percentile(if_scores, 90)  # Top 10% most anomalous
        results['dbscan_cluster'] = dbscan_labels
        results['is_outlier_dbscan'] = dbscan_labels == -1  # DBSCAN outliers
        
        # Combined risk score (0 to 1, higher means more suspicious)
        results['risk_score'] = (
            (results['is_outlier_if'].astype(int) + 
             results['is_outlier_dbscan'].astype(int)) / 2
        )
        
        return results

def process_exam_data(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    """Process exam data and extract features"""
    extractor = MouseFeatureExtractor(window_size=window_size)
    feature_windows = []
    
    # Process each exam separately
    for exam_id in df['exam_id'].unique():
        exam_data = df[df['exam_id'] == exam_id].copy()
        
        # Process each time window
        start_time = pd.to_datetime(exam_data['created_at'])
        end_time = start_time.max()
        
        current_window_start = start_time.min()
        while current_window_start < end_time:
            current_window_end = current_window_start + timedelta(seconds=window_size)
            
            # Get window data
            window_data = exam_data[
                (pd.to_datetime(exam_data['created_at']) >= current_window_start) & 
                (pd.to_datetime(exam_data['created_at']) < current_window_end)
            ]
            
            if not window_data.empty:
                # Extract features
                features = extractor.extract_features(window_data)
                features['exam_id'] = exam_id
                features['window_start'] = current_window_start
                feature_windows.append(features)
            
            current_window_start = current_window_end
    
    return pd.DataFrame(feature_windows)

def main():
    # Load data from database
    print("Loading data...")
    # TODO: Replace with your database connection
    df = pd.read_csv('synthetic_mouse_data.csv')  # For testing
    
    # Process data and extract features
    print("Extracting features...")
    features_df = process_exam_data(df)
    
    # Train anomaly detector
    print("Training anomaly detector...")
    detector = MouseBehaviorAnomalyDetector()
    detector.fit(features_df)
    
    # Get predictions
    print("Generating predictions...")
    results = detector.predict(features_df)
    
    # Save models
    print("Saving models...")
    joblib.dump(detector, 'mouse_anomaly_detector.joblib')
    
    # Save results
    results.to_csv('anomaly_detection_results.csv', index=False)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total windows analyzed: {len(results)}")
    print(f"Windows flagged as suspicious by Isolation Forest: {results['is_outlier_if'].sum()}")
    print(f"Windows flagged as suspicious by DBSCAN: {results['is_outlier_dbscan'].sum()}")
    print("\nAverage risk scores by exam:")
    print(results.groupby('exam_id')['risk_score'].mean().round(3))

if __name__ == "__main__":
    main() 