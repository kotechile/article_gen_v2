import jwt

# Configuration from User
JWT_SECRET = "5AQA63JcXaBPTdeyknRSINfBpuM562Ht"

# Anon Key from User Env
ANON_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzdXBhYmFzZSIsImlhdCI6MTc2NDYxMjY2MCwiZXhwIjo0OTIwMjg2MjYwLCJyb2xlIjoiYW5vbiJ9.4z_OjFo4hYnh1RpOVGWJYWGWW1dWfSUtKs5w06H9PYI"

# Access Token from User Log (New Token)
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmMjQ4YjdlZC1iOGRmLTQ0NjQtODU0NC04MzA0ZDdhZTRjMzAiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzcxMDIzOTMzLCJpYXQiOjE3NzEwMjAzMzMsImVtYWlsIjoia290ZWNoaWxlQGdtYWlsLmNvbSIsInBob25lIjoiIiwiYXBwX21ldGFkYXRhIjp7InByb3ZpZGVyIjoiZW1haWwiLCJwcm92aWRlcnMiOlsiZW1haWwiLCJnb29nbGUiXX0sInVzZXJfbWV0YWRhdGEiOnsiYXZhdGFyX3VybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0pDTmJUZlctTmYwbGo2djFYSXlWRXZ0NTJPVU1VdDdtNWVWLWJBNFdDdGo0REEzSkE9czk2LWMiLCJlbWFpbCI6ImtvdGVjaGlsZUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZmlyc3RfbmFtZSI6IkpvcmdlIiwiZnVsbF9uYW1lIjoiSm9yZ2UgRmVybmFuZGV6IiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tIiwibGFzdF9uYW1lIjoiRmVybmFuZGV6IiwibmFtZSI6IkpvcmdlIEZlcm5hbmRleiIsInBob25lX3ZlcmlmaWVkIjpmYWxzZSwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0pDTmJUZlctTmYwbGo2djFYSXlWRXZ0NTJPVU1VdDdtNWVWLWJBNFdDdGo0REEzSkE9czk2LWMiLCJwcm92aWRlcl9pZCI6IjExMzQ2MzM1NjU5OTgxNzE5MTczNiIsInN1YiI6IjExMzQ2MzM1NjU5OTgxNzE5MTczNiJ9LCJyb2xlIjoiYXV0aGVudGljYXRlZCIsImFhbCI6ImFhbDEiLCJhbXIiOlt7Im1ldGhvZCI6Im9hdXRoIiwidGltZXN0YW1wIjoxNzcxMDIwMzMzfV0sInNlc3Npb25faWQiOiI3YzUyMjlhMi05NDczLTRiYjctYTg4Ny1iMWI4OTNlYjFjNjQiLCJpc19hbm9ueW1vdXMiOmZhbHNlfQ.-ra9oAhhk0FGVnBM0mUYhPI0E0La1QJYpa4GbpMn6jk"

# Service Role Key for checking user existence
SERVICE_ROLE_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzdXBhYmFzZSIsImlhdCI6MTc2NDYxMjY2MCwiZXhwIjo0OTIwMjg2MjYwLCJyb2xlIjoic2VydmljZV9yb2xlIn0.S2MWuEXYpogn2l2PlzjrKnnxzdLHxgrssMKR_0XuZLM"

from supabase import create_client
import jwt
from pprint import pprint

def verify_token(name, token, secret):
    print(f"\n--- Verifying {name} ---")
    try:
        decoded = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False, "verify_aud": False})
        print(f"✅ SUCCESS: {name} signature matches JWT_SECRET")
        pprint(decoded)
        return decoded
    except Exception as e:
        print(f"❌ FAIL: {name} signature error: {e}")
        return None

def check_user(user_id):
    print(f"\n--- Checking User {user_id} ---")
    try:
        url = "https://sbcontent.giniloh.com"
        supabase = create_client(url, SERVICE_ROLE_KEY)
        # We can't query auth.users directly via client usually, but we can use admin api
        user = supabase.auth.admin.get_user_by_id(user_id)
        if user:
            print(f"✅ User found: {user.user.email}")
            print("User Object Dump:")
            pprint(user.user.__dict__)
        else:
            print("❌ User NOT found via Admin API")
    except Exception as e:
        print(f"❌ Error checking user: {e}")

if __name__ == "__main__":
    verify_token("ANON_KEY", ANON_KEY, JWT_SECRET)
    decoded = verify_token("ACCESS_TOKEN", ACCESS_TOKEN, JWT_SECRET)
    if decoded and 'sub' in decoded:
        check_user(decoded['sub'])
