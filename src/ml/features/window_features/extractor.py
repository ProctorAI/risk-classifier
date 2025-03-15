import uvicorn
import pandas as pd
import numpy as np
from typing import Dict

class WindowStateFeatureExtractor:
    def __init__(self, window_size: int = 30,
                 rapid_switch_threshold: float = 2.0,  # seconds between switches to be considered rapid
                 suspicious_resize_threshold: float = 0.8):  # ratio threshold for suspicious resizing
        """
        Initialize the window state feature extractor.
        
        Args:
            window_size: Time window in seconds
            rapid_switch_threshold: Time threshold (in seconds) to consider switches as rapid
            suspicious_resize_threshold: Screen coverage ratio threshold for suspicious resizing
        """
        self.window_size = window_size
        self.rapid_switch_threshold = rapid_switch_threshold
        self.suspicious_resize_threshold = suspicious_resize_threshold
    
    def extract_features(self, events_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract window state related features from event data.
        
        Features:
        - blur_count: Number of times window lost focus
        - tab_switch_count: Number of tab switches
        - window_resize_count: Number of window resize events
        - rapid_switch_count: Number of rapid switches (both tab and window)
        - total_blur_duration: Total time window was blurred (seconds)
        - avg_blur_duration: Average duration of blur events
        - suspicious_resize_count: Number of resizes that made window suspiciously small
        - tab_switch_frequency: Rate of tab switches per minute
        - window_switch_frequency: Rate of window state changes per minute
        - resize_frequency: Rate of window resizes per minute
        """
        features = {}
        timestamps = pd.to_datetime(events_df['created_at'])
        total_time = (timestamps.max() - timestamps.min()).total_seconds() if len(timestamps) > 0 else 0
        
        if total_time == 0:
            return {
                'blur_count': 0.0000,
                'tab_switch_count': 0.0000,
                'window_resize_count': 0.0000,
                'rapid_switch_count': 0.0000,
                'total_blur_duration': 0.0000,
                'avg_blur_duration': 0.0000,
                'suspicious_resize_count': 0.0000,
                'tab_switch_frequency': 0.0000,
                'window_switch_frequency': 0.0000,
                'resize_frequency': 0.0000
            }

        # Process window state changes
        window_states = events_df[events_df['type'] == 'window_state_change'].copy()
        if not window_states.empty:
            window_states['data'] = window_states['data'].apply(lambda x: x if isinstance(x, dict) else {})
            window_states['state'] = window_states['data'].apply(lambda x: x.get('state', ''))
            
            # Count blur events
            blur_events = window_states[window_states['state'] == 'blurred']
            focus_events = window_states[window_states['state'] == 'focused']
            features['blur_count'] = round(float(len(blur_events)), 4)
            
            # Calculate blur durations
            if len(blur_events) > 0 and len(focus_events) > 0:
                blur_starts = pd.to_datetime(blur_events['created_at'])
                focus_starts = pd.to_datetime(focus_events['created_at'])
                
                # Calculate durations between blur and next focus
                blur_durations = []
                for blur_time in blur_starts:
                    next_focus = focus_starts[focus_starts > blur_time]
                    if not next_focus.empty:
                        duration = (next_focus.iloc[0] - blur_time).total_seconds()
                        blur_durations.append(duration)
                
                if blur_durations:
                    features['total_blur_duration'] = round(float(sum(blur_durations)), 4)
                    features['avg_blur_duration'] = round(float(np.mean(blur_durations)), 4)
                else:
                    features['total_blur_duration'] = 0.0000
                    features['avg_blur_duration'] = 0.0000
            else:
                features['total_blur_duration'] = 0.0000
                features['avg_blur_duration'] = 0.0000

        # Process tab switches
        tab_switches = events_df[events_df['type'] == 'tab_switch'].copy()
        features['tab_switch_count'] = round(float(len(tab_switches)), 4)
        
        # Process window resizes
        window_resizes = events_df[events_df['type'] == 'window_resize'].copy()
        features['window_resize_count'] = round(float(len(window_resizes)), 4)
        
        if not window_resizes.empty:
            window_resizes['data'] = window_resizes['data'].apply(lambda x: x if isinstance(x, dict) else {})
            # Calculate suspicious resizes (window becomes too small)
            def is_suspicious_resize(data):
                try:
                    ratio = float(data.get('ratio', 1.0))
                    return ratio < self.suspicious_resize_threshold
                except (ValueError, TypeError):
                    return False
            
            suspicious_resizes = window_resizes['data'].apply(is_suspicious_resize)
            features['suspicious_resize_count'] = round(float(suspicious_resizes.sum()), 4)
        else:
            features['suspicious_resize_count'] = 0.0000
        
        # Calculate rapid switches
        all_switches = pd.concat([
            window_states[['created_at']],
            tab_switches[['created_at']]
        ]).sort_values('created_at')
        
        if len(all_switches) > 1:
            switch_times = pd.to_datetime(all_switches['created_at'])
            time_diffs = switch_times.diff().dt.total_seconds()
            rapid_switches = time_diffs[time_diffs <= self.rapid_switch_threshold]
            features['rapid_switch_count'] = round(float(len(rapid_switches)), 4)
        else:
            features['rapid_switch_count'] = 0.0000
        
        # Calculate frequencies (per minute)
        minutes = total_time / 60
        if minutes > 0:
            features['tab_switch_frequency'] = round(features['tab_switch_count'] / minutes, 4)
            features['window_switch_frequency'] = round(features['blur_count'] / minutes, 4)
            features['resize_frequency'] = round(features['window_resize_count'] / minutes, 4)
        else:
            features['tab_switch_frequency'] = 0.0000
            features['window_switch_frequency'] = 0.0000
            features['resize_frequency'] = 0.0000
        
        return features 