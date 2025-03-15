import os
from dotenv import load_dotenv
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
    
    return create_client(supabase_url, supabase_key)

def fetch_exam_events(exam_id: str = None) -> list:
    """Fetch events from the database
    
    Args:
        exam_id: Optional exam_id to filter by
    
    Returns:
        List of events
    """
    client = get_supabase_client()
    query = client.table('proctoring_logs').select('*')
    
    if exam_id:
        query = query.eq('exam_id', exam_id)
    
    response = query.execute()
    print(response)
    return response.data 