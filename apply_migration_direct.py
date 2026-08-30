import os
import sys
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Get database URL
db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_CONNECTION_STRING")

if not db_url:
    print("Error: DATABASE_URL not found in environment.")
    print('Usage: DATABASE_URL="postgresql://postgres:<password>@sbcontent.giniloh.com:5432/postgres" python3 apply_migration_direct.py [migration_file.sql]')
    exit(1)

migration_file = sys.argv[1] if len(sys.argv) > 1 else "migrations/add_linkedin_integration_tables.sql"
print(f"Connecting to database to execute {migration_file}...")

try:
    engine = sqlalchemy.create_engine(db_url)
    connection = engine.connect()
    print("Connection successful!")
    
    with open(migration_file, 'r') as f:
        sql = f.read()
        
    print(f"Executing migration from {migration_file}...")
    
    with connection.begin():
        connection.execute(text(sql))
        
    print("Migration executed successfully!")
    connection.close()
    
except Exception as e:
    print(f"Error executing migration: {e}")
    exit(1)
