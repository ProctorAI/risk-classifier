import uvicorn
import numpy as np
import pandas as pd
from typing import Dict

class MouseFeatureExtractor:
    def __init__(self, 
                 window_size: int = 30,
                 edge_threshold: float = 0.05,  # top/bottom 5% of screen
                 idle_threshold: float = 2.0):  # 2 seconds for idle detection
        """
        Initialize the feature extractor with configurable parameters.
        
        Args:
            window_size: Time window in seconds
            edge_threshold: Fraction of screen height to consider as edge (0.05 = top/bottom 5%)
            idle_threshold: Time in seconds to consider as idle period
        """
        self.window_size = window_size
        self.edge_threshold = edge_threshold
        self.idle_threshold = idle_threshold

    def extract_features(self, events_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from mouse movement data.
        
        Features calculated:
        - avg_norm_x, avg_norm_y: Average mouse position (0-1, normalized by screen size)
        - std_norm_x, std_norm_y: How much the mouse moves around (higher = more movement)
        - top_edge_time: Fraction of time spent in top edge_threshold of screen
        - bottom_edge_time: Fraction of time spent in bottom edge_threshold of screen
        - idle_percentage: Fraction of time with no movement (>idle_threshold seconds)
        """
        # Filter mouse events
        mouse_events = events_df[events_df['type'] == 'mouse_move'].copy()
        
        if len(mouse_events) < 1:
            return {
                'avg_norm_x': 0.5000,
                'avg_norm_y': 0.5000,
                'std_norm_x': 0.0000,
                'std_norm_y': 0.0000,
                'top_edge_time': 0.0000,
                'bottom_edge_time': 0.0000,
                'idle_percentage': 1.0000
            }
        
        # Extract coordinates and normalize
        coords = pd.json_normalize(mouse_events['data'])
        window_width = mouse_events['window_width'].iloc[0]
        window_height = mouse_events['window_height'].iloc[0]
        
        # Convert coordinates to float and normalize
        norm_x = coords['x'].astype(float) / window_width
        norm_y = coords['y'].astype(float) / window_height
        
        # Calculate basic position features
        avg_norm_y = norm_y.mean()
        features = {
            'avg_norm_x': round(norm_x.mean(), 4),
            'avg_norm_y': round(avg_norm_y, 4),
            'std_norm_x': round(norm_x.std() if len(norm_x) > 1 else 0.0, 4),
            'std_norm_y': round(norm_y.std() if len(norm_y) > 1 else 0.0, 4)
        }
        
        # Simple edge detection based on average position
        features['top_edge_time'] = round(1.0 if avg_norm_y <= self.edge_threshold else 0.0, 4)
        features['bottom_edge_time'] = round(1.0 if avg_norm_y >= (1 - self.edge_threshold) else 0.0, 4)
        
        # Calculate idle time using timestamps
        timestamps = pd.to_datetime(mouse_events['created_at'])
        time_diffs = timestamps.diff().dt.total_seconds()
        total_time = (timestamps.max() - timestamps.min()).total_seconds()
        
        if total_time > 0:
            # Calculate idle time
            idle_times = time_diffs[time_diffs > self.idle_threshold]
            features['idle_percentage'] = round(idle_times.sum() / total_time, 4)
        else:
            features['idle_percentage'] = 1.0000
            
        return features 