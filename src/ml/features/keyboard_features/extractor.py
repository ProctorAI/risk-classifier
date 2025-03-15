import uvicorn
import numpy as np
import pandas as pd
from typing import Dict, Set

class KeyboardFeatureExtractor:
    def __init__(self,
                 window_size: int = 30,
                 shortcut_keys: Set[str] = {'Control', 'Alt', 'Tab', 'Meta', 'Shift'},
                 rapid_key_threshold: float = 0.1,  # seconds between keystrokes to be considered rapid
                 backspace_burst_threshold: float = 1.0):  # seconds between backspaces to be considered a burst
        """
        Initialize the keyboard feature extractor with configurable parameters.
        
        Args:
            window_size: Time window in seconds
            shortcut_keys: Set of keys considered as shortcut/system keys
            rapid_key_threshold: Time threshold (in seconds) for rapid typing detection
            backspace_burst_threshold: Time threshold (in seconds) for backspace burst detection
        """
        self.window_size = window_size
        self.shortcut_keys = shortcut_keys
        self.rapid_key_threshold = rapid_key_threshold
        self.backspace_burst_threshold = backspace_burst_threshold
        
    def extract_features(self, events_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract keyboard and clipboard-related features from event data.
        
        Features calculated:
        - key_press_rate: Average number of keystrokes per second
        - key_press_count: Total number of keystrokes
        - shortcut_key_ratio: Ratio of shortcut keys to total keystrokes
        - shortcut_key_count: Number of shortcut keys pressed
        - alt_key_count: Number of Alt key presses
        - tab_key_count: Number of Tab key presses
        - meta_key_count: Number of Meta/Windows key presses
        - control_key_count: Number of Control key presses
        - shift_key_count: Number of Shift key presses
        - backspace_ratio: Ratio of backspace/delete keys to total keystrokes
        - backspace_count: Number of backspace/delete keys pressed
        - backspace_burst_count: Number of rapid backspace sequences
        - rapid_key_ratio: Ratio of rapid keystrokes to total keystrokes
        - rapid_key_count: Number of rapid keystrokes
        - clipboard_operation_rate: Number of clipboard operations per minute
        - clipboard_operation_count: Total number of clipboard operations
        - copy_count: Number of copy operations
        - cut_count: Number of cut operations
        - paste_count: Number of paste operations
        - avg_clipboard_length: Average length of clipboard content
        """
        features = {}
        timestamps = pd.to_datetime(events_df['created_at'])
        total_time = (timestamps.max() - timestamps.min()).total_seconds() if len(timestamps) > 0 else 0
        
        if total_time == 0:
            return {
                'key_press_rate': 0.0000,
                'key_press_count': 0.0000,
                'shortcut_key_ratio': 0.0000,
                'shortcut_key_count': 0.0000,
                'alt_key_count': 0.0000,
                'tab_key_count': 0.0000,
                'meta_key_count': 0.0000,
                'control_key_count': 0.0000,
                'shift_key_count': 0.0000,
                'backspace_ratio': 0.0000,
                'backspace_count': 0.0000,
                'backspace_burst_count': 0.0000,
                'rapid_key_ratio': 0.0000,
                'rapid_key_count': 0.0000,
                'clipboard_operation_rate': 0.0000,
                'clipboard_operation_count': 0.0000,
                'copy_count': 0.0000,
                'cut_count': 0.0000,
                'paste_count': 0.0000,
                'avg_clipboard_length': 0.0000
            }

        # Process keyboard events
        key_events = events_df[events_df['type'] == 'key_press'].copy()
        if len(key_events) > 0:
            key_types = pd.json_normalize(key_events['data'])['key_type']
            key_timestamps = pd.to_datetime(key_events['created_at'])
            time_diffs = key_timestamps.diff().dt.total_seconds()
            
            # Basic keyboard metrics
            features['key_press_count'] = round(float(len(key_events)), 4)
            features['key_press_rate'] = round(features['key_press_count'] / total_time, 4)
            
            # Individual suspicious key counts
            features['alt_key_count'] = round(float(sum(key_types == 'Alt')), 4)
            features['tab_key_count'] = round(float(sum(key_types == 'Tab')), 4)
            features['meta_key_count'] = round(float(sum(key_types == 'Meta')), 4)
            features['control_key_count'] = round(float(sum(key_types == 'Control')), 4)
            features['shift_key_count'] = round(float(sum(key_types == 'Shift')), 4)
            
            # Shortcut key patterns (keeping ratio for overall picture)
            shortcut_count = sum(1 for key in key_types if key in self.shortcut_keys)
            features['shortcut_key_count'] = round(float(shortcut_count), 4)
            features['shortcut_key_ratio'] = round(shortcut_count / len(key_events), 4)
            
            # Backspace patterns
            backspace_count = sum(1 for key in key_types if key in ['Backspace', 'Delete'])
            features['backspace_count'] = round(float(backspace_count), 4)
            features['backspace_ratio'] = round(backspace_count / len(key_events), 4)
            
            # Backspace bursts
            backspace_bursts = sum(1 for i in range(len(key_types)) 
                if key_types.iloc[i] in ['Backspace', 'Delete'] and 
                i > 0 and time_diffs.iloc[i] <= self.backspace_burst_threshold)
            features['backspace_burst_count'] = round(float(backspace_bursts), 4)
            
            # Rapid typing patterns
            rapid_keystrokes = sum(1 for t in time_diffs if t <= self.rapid_key_threshold)
            features['rapid_key_count'] = round(float(rapid_keystrokes), 4)
            features['rapid_key_ratio'] = round(rapid_keystrokes / len(key_events), 4)
        else:
            features.update({
                'key_press_rate': 0.0000,
                'key_press_count': 0.0000,
                'shortcut_key_ratio': 0.0000,
                'shortcut_key_count': 0.0000,
                'alt_key_count': 0.0000,
                'tab_key_count': 0.0000,
                'meta_key_count': 0.0000,
                'control_key_count': 0.0000,
                'shift_key_count': 0.0000,
                'backspace_ratio': 0.0000,
                'backspace_count': 0.0000,
                'backspace_burst_count': 0.0000,
                'rapid_key_ratio': 0.0000,
                'rapid_key_count': 0.0000
            })

        # Process clipboard events
        clipboard_events = events_df[events_df['type'] == 'clipboard'].copy()
        if len(clipboard_events) > 0:
            clipboard_data = pd.json_normalize(clipboard_events['data'])
            
            # Calculate clipboard operation counts and rate
            features['clipboard_operation_count'] = round(float(len(clipboard_events)), 4)
            features['clipboard_operation_rate'] = round((features['clipboard_operation_count'] * 60) / total_time, 4)
            
            # Calculate counts for each operation type
            features['copy_count'] = round(float(sum(clipboard_data['action'] == 'copy')), 4)
            features['cut_count'] = round(float(sum(clipboard_data['action'] == 'cut')), 4)
            features['paste_count'] = round(float(sum(clipboard_data['action'] == 'paste')), 4)
            
            # Calculate average selection length
            selection_lengths = clipboard_data['selection'].astype(str).str.len()
            features['avg_clipboard_length'] = round(selection_lengths.mean(), 4)
        else:
            features.update({
                'clipboard_operation_rate': 0.0000,
                'clipboard_operation_count': 0.0000,
                'copy_count': 0.0000,
                'cut_count': 0.0000,
                'paste_count': 0.0000,
                'avg_clipboard_length': 0.0000
            })
            
        return features 