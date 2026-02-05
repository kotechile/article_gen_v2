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
