
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def check_schema():
    print("Probing specific columns in Titles table...")
    target_columns = [
        'Title', 'hook', 'thesis', 'htmlArticle', 
        'featuredImageURL', 'ImageAuthor', 'mediaAltText', 'mediaTitle', 'mediaCaption',
        'citations', 'include_in_text_citations', 'selected_citations',
        'last_wp_site_id', 'last_wp_post_status', 'last_wp_category_id'
    ]
    
    for col in target_columns:
        try:
            print(f"Testing '{col}'...", end=" ")
            supabase.table('Titles').select(col).limit(1).execute()
            print("EXISTS")
        except Exception as e:
            if "column" in str(e).lower() and "not found" in str(e).lower():
                print("MISSING")
            else:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    check_schema()
