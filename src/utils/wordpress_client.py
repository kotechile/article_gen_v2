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
            
        self.username = username
        self.app_password = app_password
        
        # Create auth header
        if self.username and self.app_password:
            credentials = f"{self.username}:{self.app_password}"
            token = base64.b64encode(credentials.encode()).decode()
            self.headers = {'Authorization': f'Basic {token}'}
        else:
            self.headers = {}

    def get_posts(self, page: int = 1, per_page: int = 20) -> List[Dict]:
        """Fetch posts from WordPress site"""
        try:
            url = f"{self.base_url}/posts"
            params = {
                'page': page,
                'per_page': per_page,
                'status': 'publish',
                '_fields': 'id,title,link,excerpt' # Optimize response size
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10, verify=False)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts from {self.base_url}: {str(e)}")
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

    def create_category(self, name: str, slug: str, parent: int = 0) -> Dict:
        """Create a WordPress category."""
        try:
            url = f"{self.base_url}/categories"
            payload = {
                "name": name,
                "slug": slug,
                "parent": parent or 0,
            }
            response = requests.post(url, headers={**self.headers, 'Content-Type': 'application/json'}, json=payload, timeout=15, verify=False)
            if not response.ok:
                # Common WP behavior: term exists -> return existing category id in error payload.
                try:
                    err = response.json() or {}
                    if err.get("code") == "term_exists":
                        existing_id = ((err.get("data") or {}).get("term_id"))
                        if existing_id:
                            return self.get_category(int(existing_id))
                except Exception:
                    pass
                response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error creating category on {self.base_url}: {str(e)}")
            raise e

    def update_category(self, category_id: int, name: Optional[str] = None, slug: Optional[str] = None, parent: Optional[int] = None) -> Dict:
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

            response = requests.post(url, headers={**self.headers, 'Content-Type': 'application/json'}, json=payload, timeout=15, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error updating category {category_id} on {self.base_url}: {str(e)}")
            raise e
