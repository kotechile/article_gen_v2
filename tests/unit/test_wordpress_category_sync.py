import copy
import os
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.app import create_app


class _Result:
    def __init__(self, data):
        self.data = data


class _TableQuery:
    def __init__(self, fake_supabase, table_name: str):
        self._db = fake_supabase
        self._table_name = table_name
        self._filters = []
        self._action = "select"
        self._update_payload = {}

    def select(self, *_args, **_kwargs):
        self._action = "select"
        return self

    def eq(self, field, value):
        self._filters.append((field, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self._action = "update"
        self._update_payload = payload
        return self

    def execute(self):
        rows = self._db.tables[self._table_name]

        def matches(row):
            return all(row.get(field) == value for field, value in self._filters)

        matched = [row for row in rows if matches(row)]

        if self._action == "select":
            return _Result(copy.deepcopy(matched))

        if self._action == "update":
            for row in rows:
                if matches(row):
                    row.update(copy.deepcopy(self._update_payload))
            return _Result(copy.deepcopy(matched))

        raise NotImplementedError(f"Unsupported action: {self._action}")


class _FakeSupabase:
    def __init__(self):
        self.tables = {
            "projects": [
                {
                    "id": "project-1",
                    "user_id": "user-1",
                    "domain": "example.com",
                    "wpusername": "wp-user",
                    "wordpress_key": "wp-pass",
                    "app_name": "Example Project",
                }
            ],
            "project_categories": [
                {
                    "id": "cat-1",
                    "project_id": "project-1",
                    "user_id": "user-1",
                    "name": "Decision Engineering",
                    "description": "Updated local description",
                    "slug": "decision-engineering",
                    "level": 1,
                    "parent_category_id": None,
                    "sort_order": 1,
                    "wordpress_category_id": 101,
                    "wordpress_parent_category_id": None,
                    "wordpress_site_domain": "example.com",
                }
            ],
        }

    def table(self, table_name: str):
        return _TableQuery(self, table_name)


class _FakeWordPressClient:
    created_calls = []
    updated_calls = []

    def __init__(self, domain: str, username: str, app_password: str):
        self.domain = domain
        self.username = username
        self.app_password = app_password

    def get_categories_detailed(self, per_page: int = 100):
        return [
            {
                "id": 101,
                "name": "Old WP Name",
                "slug": "old-wp-name",
                "parent": 0,
                "count": 0,
                "description": "Old WP description",
            }
        ]

    def get_category(self, category_id: int):
        if category_id != 101:
            raise AssertionError(f"Unexpected category lookup: {category_id}")
        return {
            "id": 101,
            "name": "Old WP Name",
            "slug": "old-wp-name",
            "parent": 0,
            "description": "Old WP description",
        }

    def create_category(self, *args, **kwargs):
        self.__class__.created_calls.append((args, kwargs))
        raise AssertionError("create_category should not be called for mapped categories")

    def update_category(self, category_id: int, **kwargs):
        self.__class__.updated_calls.append((category_id, kwargs))
        return {
            "id": category_id,
            "name": kwargs.get("name"),
            "slug": kwargs.get("slug"),
            "parent": kwargs.get("parent", 0),
            "description": kwargs.get("description"),
        }


def _headers():
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-key",
        "Authorization": "Bearer test-token",
    }


def test_sync_project_categories_updates_existing_wordpress_category_by_stored_id(monkeypatch):
    from src.api import wordpress as wordpress_module

    fake_supabase = _FakeSupabase()
    _FakeWordPressClient.created_calls = []
    _FakeWordPressClient.updated_calls = []

    monkeypatch.setattr(wordpress_module, "get_supabase_client", lambda: fake_supabase)
    monkeypatch.setattr(wordpress_module, "WordPressClient", _FakeWordPressClient)
    monkeypatch.setattr(
        wordpress_module,
        "_generate_category_descriptions",
        lambda domain, project_name, categories: {
            str(category["id"]): category.get("description") or ""
            for category in categories
        },
    )

    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["API_KEYS"] = []
    client = app.test_client()

    response = client.post(
        "/api/wordpress/sync-project-categories",
        headers=_headers(),
        json={"user_id": "user-1", "project_id": "project-1"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["created"] == 0
    assert payload["updated"] == 1
    assert payload["errors_count"] == 0

    assert _FakeWordPressClient.created_calls == []
    assert _FakeWordPressClient.updated_calls == [
        (
            101,
            {
                "name": "Decision Engineering",
                "slug": "decision-engineering",
                "parent": 0,
                "description": "Updated local description",
            },
        )
    ]

    persisted = fake_supabase.tables["project_categories"][0]
    assert persisted["wordpress_category_id"] == 101
    assert persisted["wordpress_parent_category_id"] is None
    assert persisted["wordpress_site_domain"] == "example.com"


class _MissingMappedWordPressClient(_FakeWordPressClient):
    def get_categories_detailed(self, per_page: int = 100):
        return []

    def get_category(self, category_id: int):
        response = requests.Response()
        response.status_code = 404
        response._content = b'{"code":"rest_term_invalid","message":"Term not found."}'
        error = requests.exceptions.HTTPError("404 Client Error: Not Found for url", response=response)
        raise error

    def create_category(self, *args, **kwargs):
        self.__class__.created_calls.append((args, kwargs))
        return {
            "id": 202,
            "name": kwargs.get("name"),
            "slug": kwargs.get("slug"),
            "parent": kwargs.get("parent", 0),
            "description": kwargs.get("description"),
        }


def test_sync_project_categories_recreates_missing_mapped_wordpress_category_and_persists_new_id(monkeypatch):
    from src.api import wordpress as wordpress_module

    fake_supabase = _FakeSupabase()
    _MissingMappedWordPressClient.created_calls = []
    _MissingMappedWordPressClient.updated_calls = []

    monkeypatch.setattr(wordpress_module, "get_supabase_client", lambda: fake_supabase)
    monkeypatch.setattr(wordpress_module, "WordPressClient", _MissingMappedWordPressClient)
    monkeypatch.setattr(
        wordpress_module,
        "_generate_category_descriptions",
        lambda domain, project_name, categories: {
            str(category["id"]): category.get("description") or ""
            for category in categories
        },
    )

    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["API_KEYS"] = []
    client = app.test_client()

    response = client.post(
        "/api/wordpress/sync-project-categories",
        headers=_headers(),
        json={"user_id": "user-1", "project_id": "project-1"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["created"] == 1
    assert payload["updated"] == 0
    assert payload["errors_count"] == 0

    assert _MissingMappedWordPressClient.updated_calls == []
    assert _MissingMappedWordPressClient.created_calls == [
        (
            (),
            {
                "name": "Decision Engineering",
                "slug": "decision-engineering",
                "parent": 0,
                "description": "Updated local description",
            },
        )
    ]

    persisted = fake_supabase.tables["project_categories"][0]
    assert persisted["wordpress_category_id"] == 202
    assert persisted["wordpress_parent_category_id"] is None
    assert persisted["wordpress_site_domain"] == "example.com"
