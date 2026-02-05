
import os
import sys
import logging
from dotenv import load_dotenv
from supabase_client import get_supabase_client

# Load environment variables
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)


TARGET_ID_LOGS = '08227889-db30-449b-8c8b-6fc6fa4aaf60'
TARGET_ID_USER = '301d679e-80f4-4661-bf7d-be342c4fcb86'

def verify_save():
    client = get_supabase_client()
    if not client:
        print("Failed to initialize Supabase client")
        return

    for aid in [TARGET_ID_LOGS, TARGET_ID_USER]:
        print(f"\nVerifying article {aid}...")
        try:
            response = client.table('Titles').select('*').eq('id', aid).execute()
            
            if not response.data:
                print(f"Article {aid} NOT FOUND in DB.")
                continue

            article = response.data[0]
            print(f"FOUND! Status: {article.get('status')}")
            print(f"Title: {article.get('Title')}")
            
            # Check content fields
            text_len = len(article.get('articleText') or '')
            html_len = len(article.get('htmlArticle') or '')
            
            print(f"articleText length: {text_len}")
            print(f"htmlArticle length: {html_len}")
            
            if text_len > 0:
                print("SUCCESS: Content was saved.")
                print(f"Snippet: {article.get('articleText')[:100]}...")
            else:
                print("FAILURE: Content is empty.")

        except Exception as e:
            print(f"Error fetching article: {e}")

if __name__ == "__main__":
    verify_save()
