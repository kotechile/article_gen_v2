
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


def check_titles_schema():
    print("Probing Titles table columns...")
    
    client = get_supabase_client()
    if not client:
        print("Failed to initialize Supabase client")
        return

    columns_to_test = [
        'status',          # Assumed correct
        'Content',         # Known bad
        'content',         # Potential correct
        'articleText',     # Potential correct (from code usage)
        'htmlArticle',     # Potential correct
        'HTML Content',    # Known bad?
        'html_content',    # Potential correct
        'seo_optimization_score'
    ]
    
    for col in columns_to_test:
        try:
            print(f"Testing column: '{col}'...", end=" ")
            # Select specfic column
            client.table('Titles').select(col).limit(1).execute()
            print("EXISTS (or at least queryable)")
        except Exception as e:
            msg = str(e)
            if "Could not find the" in msg and "column of 'Titles'" in msg:
                 print("DOES NOT EXIST")
            else:
                 print(f"ERROR: {msg}")

if __name__ == "__main__":
    check_titles_schema()
