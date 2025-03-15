from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel
import pytz

from src.ml.utils.database import get_supabase_client
from src.ml.features.mouse_features.extractor import MouseFeatureExtractor
from src.ml.features.keyboard_features.extractor import KeyboardFeatureExtractor
from src.ml.features.window_features.extractor import WindowStateFeatureExtractor

router = APIRouter(
    prefix="/features",
    tags=["features"]
)

class FeaturesRequest(BaseModel):
    exam_id: str
    interval_seconds: int = 300  # Default 5 minutes in seconds
    fallback_limit: int = 100  # Number of most recent events to use if no events in interval

class IntervalFeatures(BaseModel):
    interval_start: datetime
    interval_end: datetime
    mouse_features: Dict
    keyboard_features: Dict
    window_features: Dict
    event_count: int  # Added to show how many events were processed

class FeaturesResponse(BaseModel):
    exam_id: str
    intervals_processed: int
    features: List[IntervalFeatures]
    used_fallback: bool  # Indicates if we used fallback data

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify the features router is working"""
    return {"status": "ok", "message": "Features router is working"}

@router.post("/extract", response_model=FeaturesResponse)
async def extract_features(request: FeaturesRequest):
    """
    Extract features from proctoring events for an exam
    
    Args:
        request: FeaturesRequest containing exam_id, interval_seconds, and fallback_limit
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        used_fallback = False
        
        # First try getting events from the specified time window
        current_time = datetime.now(pytz.UTC)
        window_start = current_time - timedelta(seconds=request.interval_seconds)
        
        # Get events for the time window
        response = supabase.table('proctoring_logs')\
            .select('*')\
            .eq('exam_id', request.exam_id)\
            .gte('created_at', window_start.isoformat())\
            .lte('created_at', current_time.isoformat())\
            .order('created_at')\
            .execute()
            
        # If no events in time window, fall back to most recent events
        if not response.data:
            used_fallback = True
            response = supabase.table('proctoring_logs')\
                .select('*')\
                .eq('exam_id', request.exam_id)\
                .order('created_at', desc=True)\
                .limit(request.fallback_limit)\
                .execute()
                
            if not response.data:
                raise HTTPException(status_code=404, detail="No events found for this exam")
        
        # Convert to DataFrame and ensure UTC timezone
        events = pd.DataFrame(response.data)
        events['created_at'] = pd.to_datetime(events['created_at']).dt.tz_convert('UTC')
        
        if used_fallback:
            # For fallback data, use the actual time range from the events
            window_start = events['created_at'].min()
            current_time = events['created_at'].max()
        
        # Initialize feature extractors
        mouse_extractor = MouseFeatureExtractor()
        keyboard_extractor = KeyboardFeatureExtractor()
        window_extractor = WindowStateFeatureExtractor()
        
        # Extract features for the interval
        mouse_features = mouse_extractor.extract_features(events)
        keyboard_features = keyboard_extractor.extract_features(events)
        window_features = window_extractor.extract_features(events)
        
        # Create response
        interval_features = IntervalFeatures(
            interval_start=window_start,
            interval_end=current_time,
            mouse_features=mouse_features,
            keyboard_features=keyboard_features,
            window_features=window_features,
            event_count=len(events)
        )
        
        return FeaturesResponse(
            exam_id=request.exam_id,
            intervals_processed=1,
            features=[interval_features],
            used_fallback=used_fallback
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 