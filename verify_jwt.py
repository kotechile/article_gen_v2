import os
import jwt

# Configuration from Environment
JWT_SECRET = os.environ.get("JWT_SECRET")

# Anon Key from Environment
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Access Token from Environment
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")

# Service Role Key from Environment
SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


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
