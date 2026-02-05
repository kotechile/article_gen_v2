import os
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def update_schema():
    if not DATABASE_URL:
        print("Error: DATABASE_URL not found in environment variables.")
        return

    print(f"Connecting to database...")
    try:
        engine = sqlalchemy.create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Running SQL updates...")
            
            # SQL statements from user request
            sql_statements = [
                "ALTER TABLE wordPress_details ADD COLUMN IF NOT EXISTS site_description TEXT;",
                "ALTER TABLE wordPress_details ADD COLUMN IF NOT EXISTS last_trend_report JSONB;",
                "ALTER TABLE wordPress_details ADD COLUMN IF NOT EXISTS target_keywords TEXT[];"
            ]
            
            for sql in sql_statements:
                print(f"Executing: {sql}")
                connection.execute(text(sql))
            
            connection.commit()
            print("Database updated successfully.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_schema()
