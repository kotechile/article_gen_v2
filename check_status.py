
import os
from dotenv import load_dotenv

load_dotenv(override=True)
from supabase_client import get_supabase_client

def check_recent_articles():
    supabase = get_supabase_client()
    if not supabase:
        print("Failed to initialize client")
        return

    # Inspect columns
    response = supabase.table('Titles').select('id, Title, status, dateCreatedOn, seo_optimization_score').order('dateCreatedOn', desc=True).limit(5).execute()
    
    print(f"Recent Articles:")
    for article in response.data:
        print(f"ID: {article['id']} | Status: {article['status']} | Created: {article['dateCreatedOn']} | Title: {article.get('Title')[:30]}")

if __name__ == "__main__":
    check_recent_articles()
