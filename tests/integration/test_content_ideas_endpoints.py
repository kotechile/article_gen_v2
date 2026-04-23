import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.api.app import create_app


class _Result:
    def __init__(self, data):
        self.data = data


class _AuthUser:
    def __init__(self, user_id: str):
        self.user = type("UserObj", (), {"id": user_id})()


class _Auth:
    def __init__(self, user_id: str):
        self._user_id = user_id

    def get_user(self, _token: str):
        return _AuthUser(self._user_id)


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

    def order(self, *_args, **_kwargs):
        return self

    def eq(self, field, value):
        self._filters.append((field, value))
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self._action = "update"
        self._update_payload = payload
        return self

    def delete(self):
        self._action = "delete"
        return self

    def insert(self, payload):
        self._action = "insert"
        self._insert_payload = payload
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
                    row.update(self._update_payload)
            return _Result(copy.deepcopy(matched))

        if self._action == "delete":
            self._db.tables[self._table_name] = [row for row in rows if not matches(row)]
            return _Result(copy.deepcopy(matched))

        if self._action == "insert":
            payload = self._insert_payload
            if self._table_name == "Titles" and self._db.titles_missing_columns:
                payload_rows = payload if isinstance(payload, list) else [payload]
                for row in payload_rows:
                    missing_hits = [col for col in self._db.titles_missing_columns if col in row]
                    if missing_hits:
                        missing_msg = " ".join(
                            [
                                f"Could not find the '{col}' column of 'Titles' in the schema cache"
                                for col in missing_hits
                            ]
                        )
                        raise Exception(missing_msg)
            if isinstance(payload, list):
                inserted = [copy.deepcopy(item) for item in payload]
                rows.extend(inserted)
                return _Result(inserted)
            inserted = copy.deepcopy(payload)
            rows.append(inserted)
            return _Result([inserted])

        return _Result([])


class _FakeSupabase:
    def __init__(self, user_id: str, rows, titles_missing_columns=None):
        self.auth = _Auth(user_id)
        self.titles_missing_columns = set(titles_missing_columns or [])
        self.tables = {
            "content_ideas": copy.deepcopy(rows),
            "Titles": [],
        }

    def table(self, table_name: str):
        return _TableQuery(self, table_name)


def _build_client(monkeypatch, rows, user_id="user-1", titles_missing_columns=None):
    from src.api.endpoints import content_ideas as content_ideas_module

    fake = _FakeSupabase(
        user_id=user_id,
        rows=rows,
        titles_missing_columns=titles_missing_columns,
    )
    monkeypatch.setattr(content_ideas_module, "get_supabase_client", lambda: fake)

    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["API_KEYS"] = []  # allow auth middleware in tests
    return app.test_client(), fake


def _headers():
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-key",
        "Authorization": "Bearer test-token",
    }


def test_list_content_ideas_filters_by_user_topic_and_type(monkeypatch):
    rows = [
        {"id": "1", "user_id": "user-1", "topic_id": "t1", "content_type": "blog", "title": "A"},
        {"id": "2", "user_id": "user-1", "topic_id": "t1", "content_type": "software", "title": "B"},
        {"id": "3", "user_id": "user-1", "topic_id": "t2", "content_type": "blog", "title": "C"},
        {"id": "4", "user_id": "user-2", "topic_id": "t1", "content_type": "blog", "title": "D"},
    ]
    client, _ = _build_client(monkeypatch, rows)

    response = client.post(
        "/api/content-ideas/list",
        headers=_headers(),
        json={"user_id": "user-1", "topic_id": "t1", "content_type": "blog"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["id"] == "1"


def test_publish_content_ideas_marks_rows_published(monkeypatch):
    rows = [
        {"id": "1", "user_id": "user-1", "status": "draft", "published": False},
        {"id": "2", "user_id": "user-1", "status": "draft", "published": False},
        {"id": "3", "user_id": "user-2", "status": "draft", "published": False},
    ]
    client, fake = _build_client(monkeypatch, rows)

    response = client.post(
        "/api/content-ideas/publish",
        headers=_headers(),
        json={"user_id": "user-1", "idea_ids": ["1", "2", "3"]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["published_count"] == 2

    persisted = {row["id"]: row for row in fake.tables["content_ideas"]}
    assert persisted["1"]["status"] == "published"
    assert persisted["1"]["published"] is True
    assert persisted["2"]["status"] == "published"
    assert persisted["2"]["published"] is True
    assert persisted["3"]["status"] == "draft"
    assert persisted["3"]["published"] is False


def test_publish_content_ideas_copies_wp_and_domain_fields_to_titles(monkeypatch):
    rows = [
        {
            "id": "1",
            "user_id": "user-1",
            "topic_id": "topic-1",
            "title": "How to choose a trail backpack",
            "description": "Decision framework for sizing and fit.",
            "content_type": "blog",
            "keywords": ["trail backpack sizing"],
            "wordpress_category_id": 22,
            "wordpress_parent_category_id": 5,
            "category": "Backpacking gear buying guides",
            "domain": "example.com",
            "status": "draft",
            "published": False,
        }
    ]
    client, fake = _build_client(monkeypatch, rows)

    response = client.post(
        "/api/content-ideas/publish",
        headers=_headers(),
        json={"user_id": "user-1", "idea_ids": ["1"]},
    )

    assert response.status_code == 200
    titles = fake.tables["Titles"]
    assert len(titles) == 1
    assert titles[0]["wordpress_category_id"] == 22
    assert titles[0]["wordpress_parent_category_id"] == 5
    assert titles[0]["category"] == "Backpacking gear buying guides"
    assert titles[0]["domain"] == "example.com"


def test_publish_content_ideas_falls_back_when_titles_columns_missing(monkeypatch):
    rows = [
        {
            "id": "1",
            "user_id": "user-1",
            "topic_id": "topic-1",
            "title": "How to choose a trail backpack",
            "description": "Decision framework for sizing and fit.",
            "content_type": "blog",
            "keywords": ["trail backpack sizing"],
            "wordpress_category_id": 22,
            "wordpress_parent_category_id": 5,
            "category": "Backpacking gear buying guides",
            "domain": "example.com",
            "status": "draft",
            "published": False,
        }
    ]
    client, fake = _build_client(
        monkeypatch,
        rows,
        titles_missing_columns={"keyword_candidates_json"},
    )

    response = client.post(
        "/api/content-ideas/publish",
        headers=_headers(),
        json={"user_id": "user-1", "idea_ids": ["1"]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["published_to_titles_count"] == 1

    titles = fake.tables["Titles"]
    assert len(titles) == 1
    assert "keyword_candidates_json" not in titles[0]
    assert titles[0]["wordpress_category_id"] == 22
    assert titles[0]["wordpress_parent_category_id"] == 5
    assert titles[0]["category"] == "Backpacking gear buying guides"
    assert titles[0]["domain"] == "example.com"


def test_delete_content_idea_requires_owner(monkeypatch):
    rows = [
        {"id": "1", "user_id": "user-1", "title": "Mine"},
        {"id": "2", "user_id": "user-2", "title": "Other"},
    ]
    client, fake = _build_client(monkeypatch, rows)

    ok_response = client.delete("/api/content-ideas/1?user_id=user-1", headers=_headers())
    assert ok_response.status_code == 200

    not_found_response = client.delete("/api/content-ideas/2?user_id=user-1", headers=_headers())
    assert not_found_response.status_code == 404

    remaining_ids = {row["id"] for row in fake.tables["content_ideas"]}
    assert remaining_ids == {"2"}
