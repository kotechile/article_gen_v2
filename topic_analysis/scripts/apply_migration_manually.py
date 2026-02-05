import os
import psycopg2
from urllib.parse import urlparse

# URL extracted from backend/.env
DATABASE_URL = "postgresql://postgres:cV3dWDPG8U73tIajmm1f2Hm9Wp8KlRoJ@sbcontent.giniloh.com:5432/postgres?sslmode=require"

MIGRATION_FILE = "backend/migrations/enrich_titles_schema.sql"

def apply_migration():
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True # Important for some DDL, though DO block is fine
        cur = conn.cursor()
        
        print(f"Reading migration file: {MIGRATION_FILE}")
        with open(MIGRATION_FILE, 'r') as f:
            sql = f.read()
            
        print("Executing SQL...")
        cur.execute(sql)
        
        print("Migration applied successfully!")
        
        # Verify columns exist
        verify_sql = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'Titles' 
        AND column_name IN ('topic_id', 'source_idea_id', 'subtopic', 'total_search_volume', 'avg_keyword_difficulty', 'content_outline')
        """
        cur.execute(verify_sql)
        rows = cur.fetchall()
        print("\nVerified columns:")
        for row in rows:
            print(f"- {row[0]}")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error applying migration: {e}")
        exit(1)

if __name__ == "__main__":
    apply_migration()
