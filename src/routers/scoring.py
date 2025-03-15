from fastapi import APIRouter, HTTPException
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel
import pytz

from src.ml.utils.database import get_supabase_client
from src.ml.features.mouse_features.extractor import MouseFeatureExtractor
from src.ml.features.keyboard_features.extractor import KeyboardFeatureExtractor
from src.ml.features.window_features.extractor import WindowStateFeatureExtractor
from src.ml.features.scoring import calculate_total_score, get_risk_level

router = APIRouter(
    prefix="/scoring",
)

def get_utc_now() -> datetime:
    """Get current UTC time with timezone info"""
    return datetime.now(pytz.UTC)

# Request Models
class RiskScoreRequest(BaseModel):
    exam_id: str
    interval_seconds: int = 300  # Default 5 minutes in seconds
    window_size_seconds: int = 900  # Default 15 minutes rolling window

class RiskScore(BaseModel):
    interval_start: datetime
    interval_end: datetime
    risk_score: float
    risk_level: str
    mouse_score: float
    keyboard_score: float
    window_score: float

class RiskScoreResponse(BaseModel):
    exam_id: str
    intervals_processed: int
    risk_scores: List[RiskScore]

def extract_features_for_interval(events_df: pd.DataFrame, start_time: datetime, end_time: datetime) -> Dict:
    """Extract features for a specific time interval"""
    # Convert datetime to UTC timestamp for comparison
    start_ts = pd.Timestamp(start_time).tz_convert('UTC')
    end_ts = pd.Timestamp(end_time).tz_convert('UTC')
    
    interval_events = events_df[
        (events_df['created_at'] >= start_ts) & 
        (events_df['created_at'] < end_ts)
    ]
    
    # If no events in interval, return default features
    if len(interval_events) == 0:
        return {
            # Mouse features (30% of total score)
            'avg_norm_x': 0,
            'avg_norm_y': 0,
            'std_norm_x': 0,
            'std_norm_y': 0,
            'top_edge_time': 0,
            'bottom_edge_time': 0,
            'idle_percentage': 0,
            
            # Keyboard features (35% of total score)
            'key_press_rate': 0,
            'shortcut_key_ratio': 0,
            'backspace_ratio': 0,
            'rapid_key_ratio': 0,
            'clipboard_operation_rate': 0,
            
            # Window state features (35% of total score)
            'blur_count': 0,
            'tab_switch_count': 0,
            'total_blur_duration': 0,
            'rapid_switch_count': 0,
            'suspicious_resize_count': 0
        }
    
    # Initialize feature extractors
    mouse_extractor = MouseFeatureExtractor()
    keyboard_extractor = KeyboardFeatureExtractor()
    window_extractor = WindowStateFeatureExtractor()
    
    # Extract features
    mouse_features = mouse_extractor.extract_features(interval_events)
    keyboard_features = keyboard_extractor.extract_features(interval_events)
    window_features = window_extractor.extract_features(interval_events)
    
    # Combine all features
    features = {**mouse_features, **keyboard_features, **window_features}
    
    # Ensure all required features exist with defaults
    required_features = {
        # Mouse features (30% of total score)
        'avg_norm_x': 0,
        'avg_norm_y': 0,
        'std_norm_x': 0,
        'std_norm_y': 0,
        'top_edge_time': 0,
        'bottom_edge_time': 0,
        'idle_percentage': 0,
        
        # Keyboard features (35% of total score)
        'key_press_rate': 0,
        'shortcut_key_ratio': 0,
        'backspace_ratio': 0,
        'rapid_key_ratio': 0,
        'clipboard_operation_rate': 0,
        
        # Window state features (35% of total score)
        'blur_count': 0,
        'tab_switch_count': 0,
        'total_blur_duration': 0,
        'rapid_switch_count': 0,
        'suspicious_resize_count': 0
    }
    
    # Update with any missing features
    for feature, default in required_features.items():
        if feature not in features:
            features[feature] = default
    
    return features

@router.post("/calculate", response_model=RiskScoreResponse)
async def calculate_risk_scores(request: RiskScoreRequest):
    """
    Calculate risk scores for an exam using a rolling window
    
    Args:
        request: RiskScoreRequest containing exam_id, interval_seconds, and window_size_seconds
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Calculate time window with UTC timezone
        current_time = get_utc_now()
        window_start = current_time - timedelta(seconds=request.window_size_seconds)
        
        # Get events only for the rolling window
        response = supabase.table('proctoring_logs')\
            .select('*')\
            .eq('exam_id', request.exam_id)\
            .gte('created_at', window_start.isoformat())\
            .lte('created_at', current_time.isoformat())\
            .order('created_at')\
            .execute()
            
        if not response.data:
            raise HTTPException(status_code=404, detail="No events found in the specified time window")
        
        # Convert to DataFrame and ensure UTC timezone
        events = pd.DataFrame(response.data)
        events['created_at'] = pd.to_datetime(events['created_at']).dt.tz_convert('UTC')
        
        # Process intervals within the window
        updates = []
        interval_start = window_start
        
        while interval_start < current_time:
            interval_end = min(interval_start + timedelta(seconds=request.interval_seconds), current_time)
            
            try:
                # Extract features for this interval
                features = extract_features_for_interval(events, interval_start, interval_end)
                
                # Calculate risk scores
                total_score, category_scores = calculate_total_score(features)
                risk_level = get_risk_level(total_score)
                
                # Update records in this interval
                update_data = {
                    'risk_score': total_score,
                    'risk_level': risk_level,
                    'mouse_score': category_scores['mouse_score'],
                    'keyboard_score': category_scores['keyboard_score'],
                    'window_score': category_scores['window_score']
                }
                
                # Convert timestamps to ISO format strings for Supabase
                start_iso = interval_start.isoformat()
                end_iso = interval_end.isoformat()
                
                # Update all records in the interval
                supabase.table('proctoring_logs')\
                    .update(update_data)\
                    .gte('created_at', start_iso)\
                    .lt('created_at', end_iso)\
                    .eq('exam_id', request.exam_id)\
                    .execute()
                
                updates.append(RiskScore(
                    interval_start=interval_start,
                    interval_end=interval_end,
                    risk_score=total_score,
                    risk_level=risk_level,
                    **category_scores
                ))
            except Exception as interval_error:
                # Log the error but continue processing other intervals
                print(f"Error processing interval {interval_start} to {interval_end}: {str(interval_error)}")
            
            interval_start = interval_end
        
        if not updates:
            raise HTTPException(status_code=500, detail="Failed to process any intervals successfully")
        
        return RiskScoreResponse(
            exam_id=request.exam_id,
            intervals_processed=len(updates),
            risk_scores=updates
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/{exam_id}")
async def get_exam_risk_summary(exam_id: str):
    """Get risk score summary for an exam"""
    try:
        supabase = get_supabase_client()
        
        # Get all scored events for this exam
        response = supabase.table('proctoring_logs')\
            .select('risk_level, risk_score, mouse_score, keyboard_score, window_score')\
            .eq('exam_id', exam_id)\
            .not_.is_('risk_score', 'null')\
            .execute()
            
        if not response.data:
            raise HTTPException(status_code=404, detail="No risk scores found for this exam")
            
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(response.data)
        
        # Calculate summary statistics
        summary = []
        for level in df['risk_level'].unique():
            level_data = df[df['risk_level'] == level]
            summary.append({
                'risk_level': level,
                'interval_count': len(level_data),
                'avg_risk_score': round(level_data['risk_score'].mean(), 4),
                'max_risk_score': round(level_data['risk_score'].max(), 4),
                'avg_mouse_score': round(level_data['mouse_score'].mean(), 4),
                'avg_keyboard_score': round(level_data['keyboard_score'].mean(), 4),
                'avg_window_score': round(level_data['window_score'].mean(), 4)
            })
        
        return {
            'exam_id': exam_id,
            'risk_summary': summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 