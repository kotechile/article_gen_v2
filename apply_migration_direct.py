import os
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# Get database URL
db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_CONNECTION_STRING")

if not db_url:
    print("Error: DATABASE_URL not found in environment.")
    exit(1)

print(f"Connecting to database...")

try:
    # Try connecting
    engine = sqlalchemy.create_engine(db_url)
    connection = engine.connect()
    print("Connection successful!")
    
    # Read the migration file
    migration_file = "migrations/create_images_table.sql"
    with open(migration_file, 'r') as f:
        sql = f.read()
        
    print(f"Executing migration from {migration_file}...")
    
    # Execute the SQL
    # We might need to split by commands if sqlalchemy doesn't handle multiple statements well in one go,
    # but usually it does or we can split by semicolon.
    # However, for DDL with $$ blocks, splitting by ; is dangerous.
    # Let's try executing as a single block first.
    
    # Wrap in transaction
    with connection.begin():
        connection.execute(text(sql))
        
    print("Migration executed successfully!")
    connection.close()
    
except Exception as e:
    print(f"Error executing migration: {e}")
    exit(1)
