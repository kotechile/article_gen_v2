
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env")
    exit(1)

def fix_schema():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()
        
        print("🔍 Checking 'Titles' table for 'created_at'...")
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'Titles' AND column_name = 'created_at'")
        if not cur.fetchone():
            print("⚠️ 'created_at' column missing in 'Titles'. Adding it...")
            cur.execute('ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();')
            print("✅ Added 'created_at' to 'Titles'")
        else:
            print("✅ 'created_at' exists in 'Titles'")

        print("🔍 Checking 'affiliate_programs' table for 'network_name'...")
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'affiliate_programs' AND column_name = 'network_name'")
        if not cur.fetchone():
            print("⚠️ 'network_name' column missing in 'affiliate_programs'. Adding it...")
            cur.execute("ALTER TABLE affiliate_programs ADD COLUMN IF NOT EXISTS network_name TEXT;")
            print("✅ Added 'network_name' to 'affiliate_programs'")
        else:
            print("✅ 'network_name' exists in 'affiliate_programs'")

        # Force schema cache reload
        print("🔄 Notifying PostgREST to reload schema cache...")
        cur.execute("NOTIFY pgrst, 'reload schema';")
        print("✅ Schema reload notification sent.")

        cur.close()
        conn.close()
        print("🎉 Fix script completed successfully.")
        
    except Exception as e:
        print(f"❌ Error during schema fix: {e}")

if __name__ == "__main__":
    fix_schema()
