
import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def check_article(article_id):
    print(f"Checking article: {article_id}")
    try:
        res = supabase.table('Titles').select('id, hook, thesis, Title').eq('id', article_id).execute()
        if res.data:
            row = res.data[0]
            print(f"Title: {row.get('Title')}")
            print(f"Hook: '{row.get('hook')}'")
            print(f"Thesis: '{row.get('thesis')}'")
        else:
            print("Article not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Article ID from logs
    check_article("fc373772-e3cc-43cf-a2fc-e380e1002aa6")
