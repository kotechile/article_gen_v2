import requests
import base64
from typing import List, Dict, Optional
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WordPressClient:
    def __init__(self, domain: str, username: str, app_password: str):
        # Ensure domain has protocol
        if not domain.startswith(('http://', 'https://')):
            self.base_url = f"https://{domain}/wp-json/wp/v2"
        else:
            self.base_url = f"{domain}/wp-json/wp/v2"
            
        self.username = (username or "").strip()
        self.app_password = (app_password or "").strip()
        
        # Create auth header
        if self.username and self.app_password:
            credentials = f"{self.username}:{self.app_password}"
            token = base64.b64encode(credentials.encode()).decode()
            self.headers = {'Authorization': f'Basic {token}'}
        else:
            self.headers = {}

    def get_current_user(self) -> Dict:
        """Fetch current authenticated user to verify credentials."""
        try:
            url = f"{self.base_url}/users/me"
            response = requests.get(url, headers=self.headers, timeout=15, verify=False)
            if not response.ok:
                try:
                    err = response.json() or {}
                    msg = err.get("message")
                    code = err.get("code")
                    if msg:
                        raise Exception(f"WordPress error ({response.status_code} {code}): {msg}")
                except Exception as inner_e:
                    if "WordPress error" in str(inner_e):
                        raise inner_e
                response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error authenticating with WordPress on {self.base_url}: {str(e)}")
            raise e

    def verify_and_optimize_auth(self) -> Dict:
        """Verify authentication with /users/me, trying variations (with/without spaces) if needed."""
        try:
            return self.get_current_user()
        except Exception as first_err:
            if not self.username or not self.app_password:
                raise first_err

            # If password had spaces, try stripped; if stripped, try with spaces
            clean_pass = self.app_password.replace(" ", "")
            if clean_pass != self.app_password:
                try:
                    alt_token = base64.b64encode(f"{self.username}:{clean_pass}".encode()).decode()
                    orig_headers = dict(self.headers)
                    self.headers = {'Authorization': f'Basic {alt_token}'}
                    user_data = self.get_current_user()
                    self.app_password = clean_pass
                    return user_data
                except Exception:
                    self.headers = orig_headers
            raise first_err

    def get_posts(self, page: int = 1, per_page: int = 20, embed: bool = True, fields: Optional[str] = None) -> List[Dict]:
        """Fetch posts from WordPress site with full SEO metadata and embedded media/terms."""
        try:
            url = f"{self.base_url}/posts"
            params: Dict = {
                'page': page,
                'per_page': per_page,
                'status': 'publish',
            }
            if embed:
                params['_embed'] = '1'
            if fields:
                params['_fields'] = fields
            
            response = requests.get(url, headers=self.headers, params=params, timeout=20, verify=False)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts from {self.base_url}: {str(e)}")
            raise e

    def get_post(self, post_id: int, embed: bool = True) -> Dict:
        """Fetch a single WordPress post by ID with full metadata."""
        try:
            url = f"{self.base_url}/posts/{post_id}"
            params = {'_embed': '1'} if embed else {}
            response = requests.get(url, headers=self.headers, params=params, timeout=15, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching post {post_id} from {self.base_url}: {str(e)}")
            raise e

    def get_categories(self) -> List[str]:
        """Fetch categories from WordPress site"""
        try:
            url = f"{self.base_url}/categories"
            params = {
                'per_page': 100,
                'hide_empty': False,
                '_fields': 'name'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10, verify=False)
            response.raise_for_status()
            
            return [cat.get('name') for cat in response.json()]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching categories from {self.base_url}: {str(e)}")
            raise e

    def get_categories_detailed(self, per_page: int = 100) -> List[Dict]:
        """Fetch detailed categories from WordPress site with pagination."""
        try:
            page = 1
            categories: List[Dict] = []

            while True:
                url = f"{self.base_url}/categories"
                params = {
                    'per_page': per_page,
                    'page': page,
                    'hide_empty': False,
                    '_fields': 'id,name,slug,parent,count,description',
                }

                response = requests.get(url, headers=self.headers, params=params, timeout=15, verify=False)
                response.raise_for_status()

                page_items = response.json() or []
                categories.extend(page_items)

                if len(page_items) < per_page:
                    break
                page += 1

            return categories
        except requests.exceptions.RequestException as e:
            print(f"Error fetching detailed categories from {self.base_url}: {str(e)}")
            raise e

    def get_category(self, category_id: int) -> Dict:
        """Fetch a single WordPress category by ID."""
        try:
            url = f"{self.base_url}/categories/{category_id}"
            response = requests.get(url, headers=self.headers, timeout=15, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching category {category_id} from {self.base_url}: {str(e)}")
            raise e

    def create_category(self, name: str, slug: str, parent: int = 0, description: Optional[str] = None) -> Dict:
        """Create a WordPress category."""
        try:
            url = f"{self.base_url}/categories"
            payload = {
                "name": name,
                "slug": slug,
                "parent": parent or 0,
            }
            if description is not None:
                payload["description"] = description
            response = requests.post(url, headers={**self.headers, 'Content-Type': 'application/json'}, json=payload, timeout=15, verify=False)
            if not response.ok:
                # Common WP behavior: term exists -> return existing category id in error payload.
                try:
                    err = response.json() or {}
                    if err.get("code") == "term_exists":
                        data = err.get("data") or {}
                        existing_id = data.get("term_id") or data.get("resource_id") or (data if isinstance(data, int) else None)
                        if existing_id:
                            return self.get_category(int(existing_id))
                    msg = err.get("message")
                    code = err.get("code")
                    if msg:
                        raise Exception(f"WordPress error ({response.status_code} {code}): {msg}")
                except Exception as inner_e:
                    if "WordPress error" in str(inner_e):
                        raise inner_e
                response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error creating category on {self.base_url}: {str(e)}")
            raise e

    def update_category(
        self,
        category_id: int,
        name: Optional[str] = None,
        slug: Optional[str] = None,
        parent: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Dict:
        """Update a WordPress category."""
        try:
            url = f"{self.base_url}/categories/{category_id}"
            payload: Dict = {}
            if name is not None:
                payload["name"] = name
            if slug is not None:
                payload["slug"] = slug
            if parent is not None:
                payload["parent"] = parent
            if description is not None:
                payload["description"] = description

            response = requests.post(url, headers={**self.headers, 'Content-Type': 'application/json'}, json=payload, timeout=15, verify=False)
            if not response.ok:
                try:
                    err = response.json() or {}
                    msg = err.get("message")
                    code = err.get("code")
                    if msg:
                        raise Exception(f"WordPress error ({response.status_code} {code}): {msg}")
                except Exception as inner_e:
                    if "WordPress error" in str(inner_e):
                        raise inner_e
                response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error updating category {category_id} on {self.base_url}: {str(e)}")
            raise e

