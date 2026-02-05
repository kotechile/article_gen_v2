
import os
import sys
from dotenv import load_dotenv

# Force load .env
load_dotenv(override=True)

from supabase_client import get_supabase_client

def find_and_reset_stuck_articles():
    supabase = get_supabase_client()
    if not supabase:
        print("Failed to initialize Supabase client")
        return

    print("Checking for stuck articles (Status: 'Generating')...")
    try:
        response = supabase.table('Titles').select('id, Title, dateCreatedOn, status').eq('status', 'Generating').execute()
        articles = response.data
        
        if not articles:
            print("No articles found with status 'Generating'.")
            return

        print(f"Found {len(articles)} stuck article(s):")
        for art in articles:
            print(f"- ID: {art['id']} | Title: {art.get('Title')} | Created: {art.get('dateCreatedOn')}")
        
        # Reset them
        for art in articles:
            print(f"Resetting article {art['id']} to 'Error'...")
            supabase.table('Titles').update({'status': 'Error'}).eq('id', art['id']).execute()
            print("Done.")
            
    except Exception as e:
        print(f"Error checking/resetting articles: {e}")

if __name__ == "__main__":
    find_and_reset_stuck_articles()
