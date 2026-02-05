import sys
from sqlalchemy import create_engine, inspect

# Try connecting to 'postgres' database via tunnel
db_url = "postgresql://postgres:cV3dWDPG8U73tIajmm1f2Hm9Wp8KlRoJ@localhost:54320/postgres"

try:
    print(f"Attempting to connect to {db_url}...")
    engine = create_engine(db_url)
    connection = engine.connect()
    print("Connection successful!")
    
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema='public')
    
    print(f"Found {len(tables)} tables:")
    for table in tables:
        print(f"- {table}")
        
    connection.close()
    
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
