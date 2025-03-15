import uvicorn
import numpy as np
from typing import Dict, Tuple

def normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a value to be between 0 and 1"""
    if value < min_val:
        return 0.0000
    if value > max_val:
        return 1.0000
    return round((value - min_val) / (max_val - min_val), 4)

def calculate_mouse_score(features: Dict[str, float]) -> float:
    """
    Calculate mouse behavior risk score (0-100)
    Higher score indicates more suspicious behavior
    
    Edge time is the primary indicator:
    - Any edge time over 1s is highly suspicious
    - Other factors matter much less
    """
    # Edge time is most suspicious - treat it very strictly
    edge_time = features['top_edge_time'] + features['bottom_edge_time']
    edge_score = 1.0000 if edge_time > 1.0 else round(edge_time, 4)  # Binary threshold at 1 second
    
    # Movement variance and idle time are secondary factors
    movement_variance = normalize(
        features['std_norm_x'] + features['std_norm_y'],
        min_val=0,
        max_val=1.0
    )
    
    idle_time = normalize(
        features['idle_percentage'],
        min_val=0,
        max_val=100
    )
    
    # Calculate weighted score with dominant edge time weight
    score = round(
        70 * edge_score +          # Edge time is the primary factor
        20 * movement_variance +    # Movement patterns less important
        10 * idle_time,            # Idle time least important
        4
    )
    
    return min(100.0000, score)

def calculate_keyboard_score(features: Dict[str, float]) -> float:
    """
    Calculate keyboard behavior risk score (0-100)
    Higher score indicates more suspicious behavior
    Focus on shortcuts and system keys
    """
    # Calculate suspicious key combinations
    suspicious_keys = (
        features.get('alt_key_count', 0) +
        features.get('tab_key_count', 0) +
        features.get('control_key_count', 0) +
        features.get('meta_key_count', 0) +
        features.get('shift_key_count', 0)
    )
    
    shortcut_score = normalize(
        suspicious_keys,
        min_val=0,
        max_val=5  # 5 or more suspicious keys in a window is maximum score
    )
    
    # Other keyboard metrics
    clipboard_rate = normalize(
        features['clipboard_operation_rate'],
        min_val=0,
        max_val=5  # Lowered threshold - 5 operations per minute is suspicious
    )
    
    rapid_typing = normalize(
        features['rapid_key_ratio'],
        min_val=0,
        max_val=0.7
    )
    
    backspace_usage = normalize(
        features['backspace_ratio'],
        min_val=0,
        max_val=0.3
    )
    
    # Calculate weighted score with emphasis on shortcuts and clipboard
    score = round(
        40 * shortcut_score +     # Suspicious key combinations most important
        35 * clipboard_rate +      # Clipboard operations very suspicious
        15 * rapid_typing +       # Rapid typing less important
        10 * backspace_usage,     # Backspace least important
        4
    )
    
    return min(100.0000, score)

def calculate_window_score(features: Dict[str, float]) -> float:
    """
    Calculate window behavior risk score (0-100)
    Higher score indicates more suspicious behavior
    Focus on rapid switches and blur duration
    """
    # Normalize key metrics with stricter thresholds
    blur_duration = normalize(
        features['total_blur_duration'],
        min_val=0,
        max_val=10  # Reduced from 20s to 10s - stricter threshold
    )
    
    rapid_switches = normalize(
        features['rapid_switch_count'],
        min_val=0,
        max_val=3  # Reduced from 5 to 3 - stricter threshold
    )
    
    tab_switches = normalize(
        features['tab_switch_count'],
        min_val=0,
        max_val=5  # Reduced from 10 to 5 - stricter threshold
    )
    
    suspicious_resizes = normalize(
        features['suspicious_resize_count'],
        min_val=0,
        max_val=2  # Reduced from 3 to 2 - stricter threshold
    )
    
    # Calculate weighted score with higher emphasis on switches
    score = round(
        35 * blur_duration +      # Blur duration very important
        35 * rapid_switches +     # Rapid switches equally important
        20 * tab_switches +       # Tab switches secondary
        10 * suspicious_resizes,  # Resizes least important
        4
    )
    
    return min(100.0000, score)

def calculate_total_score(features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total risk score and individual category scores
    
    Returns:
        Tuple of (total_score, category_scores_dict)
    """
    # Calculate individual scores
    mouse_score = calculate_mouse_score(features)
    keyboard_score = calculate_keyboard_score(features)
    window_score = calculate_window_score(features)
    
    # Calculate weighted total with higher weight for keyboard and window behavior
    total_score = round(
        0.25 * mouse_score +      # Reduced mouse weight
        0.25 * keyboard_score +   # Increased keyboard weight
        0.50 * window_score,      # Maintained window weight
        4
    )
    
    # Return scores
    return total_score, {
        'mouse_score': round(mouse_score, 4),
        'keyboard_score': round(keyboard_score, 4),
        'window_score': round(window_score, 4)
    }

def get_risk_level(score: float) -> str:
    """Convert numerical score to risk level"""
    score = round(score, 4)
    if score <= 25.0000:          # Reduced threshold for low risk
        return 'low'
    elif score <= 60.0000:        # Reduced threshold for medium risk
        return 'medium'
    else:
        return 'high' 