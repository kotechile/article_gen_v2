import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not url or not key:
    print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY/SUPABASE_KEY")
    exit(1)

supabase: Client = create_client(url, key)

REQUIRED_SCHEMA = {
    "Titles": [
        "deck", 
        "citations", 
        "include_in_text_citations", 
        "selected_citations", 
        "last_wp_site_id", 
        "last_wp_post_status", 
        "last_wp_category_id"
    ],
    "images": [
        "id", "user_id", "imageurl", "imageauthor", "mediaalttext", 
        "mediatitle", "mediacaption", "created_at"
    ],
    "llm_providers": [
        "id", "name", "provider", "model_name", "api_keys_id", 
        "base_url", "is_active", "is_default"
    ],
    "llm_providers_image": [
        "id", "model_name", "display_name", "provider", "api_keys_id"
    ],
    "wordPress_details": [
        "id", "user_id", "domain", "wpUserName", "wordpress_key", "categories"
    ],
    "wordpress_imported_posts": [
        "user_id", "wordpress_detail_id", "post_id", "title", "link", "excerpt"
    ],
}

def verify_schema():
    print("Starting Schema Verification...\n")
    
    missing_items = {}

    for table, columns in REQUIRED_SCHEMA.items():
        print(f"Checking table '{table}'...", end=" ")
        try:
            # Check table existence by selecting count
            supabase.table(table).select("count", count="exact").limit(0).execute()
            print("EXISTS")
        except Exception as e:
            print(f"MISSING ({str(e)})")
            missing_items[table] = {"status": "MISSING_TABLE", "error": str(e)}
            continue

        if columns:
            missing_cols = []
            for col in columns:
                try:
                    # Check column existence by selecting it
                    # Note: expecting empty list if no rows, but error if col invalid
                    supabase.table(table).select(col).limit(1).execute()
                except Exception as e:
                    # Typical error for missing column in PostgREST: 
                    # {'code': 'PGRST301', 'details': None, 'hint': None, 'message': 'Could not find the column ...'}
                    # or similar
                    if "column" in str(e).lower() or "find" in str(e).lower():
                         missing_cols.append(col)
                    else:
                        print(f"  Error checking col '{col}': {e}")
            
            if missing_cols:
                print(f"  MISSING COLUMNS: {', '.join(missing_cols)}")
                missing_items[table] = {"status": "MISSING_COLUMNS", "columns": missing_cols}
            else:
                print(f"  All {len(columns)} checked columns found.")

    print("\n--- Summary of Missing Items ---")
    if missing_items:
        for table, details in missing_items.items():
            if details["status"] == "MISSING_TABLE":
                print(f"Table '{table}': MISSING")
            else:
                print(f"Table '{table}': Missing Columns -> {details['columns']}")
    else:
         print("All verifications passed!")

if __name__ == "__main__":
    verify_schema()
