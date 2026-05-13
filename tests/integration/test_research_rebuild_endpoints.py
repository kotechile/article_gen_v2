import copy
import os
import sys

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
        self._insert_payload = None

    def select(self, *_args, **_kwargs):
        self._action = "select"
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
            updated = []
            for row in rows:
                if matches(row):
                    row.update(self._update_payload)
                    updated.append(copy.deepcopy(row))
            return _Result(updated)

        if self._action == "insert":
            payload = copy.deepcopy(self._insert_payload)
            if isinstance(payload, list):
                rows.extend(payload)
                return _Result(payload)
            rows.append(payload)
            return _Result([payload])

        return _Result([])


class _FakeSupabase:
    def __init__(self):
        self.tables = {
            "content_ideas": [],
            "released_software_ideas": [],
        }

    def table(self, table_name: str):
        return _TableQuery(self, table_name)


class _FakeGenerationService:
    def __init__(self, outcome):
        self.outcome = copy.deepcopy(outcome)

    async def get_record(self, *, record_id, user_id):
        return copy.deepcopy(self.outcome)

    async def update_record(self, *, record_id, user_id, data):
        self.outcome.update(data)
        return copy.deepcopy(self.outcome)


class _FakeCandidateService:
    def __init__(self, candidate):
        self.candidate = copy.deepcopy(candidate)

    async def get_record(self, *, record_id, user_id):
        return copy.deepcopy(self.candidate)


class _FakeKeywordPackService:
    def __init__(self):
        self.calls = []

    async def list_keyword_packs(self, *, user_id, project_id, candidate_id):
        self.calls.append(
            {
                "user_id": str(user_id),
                "project_id": str(project_id),
                "candidate_id": str(candidate_id),
            }
        )
        return []


class _FakeCompatibilityAdapter:
    async def outcome_to_content_idea_payload(self, *, candidate, generated_outcome, category_context, keyword_pack=None):
        return {
            "id": str(generated_outcome.get("id")),
            "title": generated_outcome.get("outcome_metadata", {}).get("title"),
            "description": generated_outcome.get("outcome_metadata", {}).get("description"),
            "content_type": "software",
            "category": "software_tool",
            "subtopic": candidate.get("candidate_text"),
            "topic_id": "topic-1",
            "keywords": [],
            "primary_keywords": [],
            "secondary_keywords": [],
            "search_phrase": generated_outcome.get("outcome_metadata", {}).get("title"),
            "idea_metadata": {
                "category_context": category_context,
            },
        }


def _headers():
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-key",
        "Authorization": "Bearer test-token",
    }


def test_release_generated_software_outcome_tolerates_missing_project_id(monkeypatch):
    from src.api.endpoints import research_rebuild as research_rebuild_module

    user_id = "33333333-3333-3333-3333-333333333333"
    fake_supabase = _FakeSupabase()
    fake_generation = _FakeGenerationService(
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "project_id": None,
            "candidate_id": "22222222-2222-2222-2222-222222222222",
            "content_idea_id": None,
            "outcome_type": "software",
            "status": "generated",
            "outcome_metadata": {
                "title": "Vacation Budget Simulator",
                "description": "Calculator for estimating the cost impact of vacations.",
                "build_complexity": "medium",
            },
        }
    )
    fake_candidate = _FakeCandidateService(
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "candidate_text": "vacation budget calculator",
            "candidate_metadata": {
                "category_context": {
                    "category_path": "Finance / Travel",
                }
            },
        }
    )
    fake_keyword_packs = _FakeKeywordPackService()

    monkeypatch.setattr(research_rebuild_module, "_get_user_id_from_request", lambda: user_id)
    monkeypatch.setattr(research_rebuild_module, "_get_admin_supabase_client", lambda: fake_supabase)
    monkeypatch.setattr(research_rebuild_module, "generation_service", fake_generation)
    monkeypatch.setattr(research_rebuild_module, "candidate_service", fake_candidate)
    monkeypatch.setattr(research_rebuild_module, "keyword_pack_service", fake_keyword_packs)
    monkeypatch.setattr(research_rebuild_module, "compatibility_adapter_service", _FakeCompatibilityAdapter())

    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["API_KEYS"] = []
    client = app.test_client()

    response = client.post(
        "/api/research-rebuild/generated-outcomes/11111111-1111-1111-1111-111111111111/release-software",
        headers=_headers(),
        json={},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["released_software_idea"]["source_idea_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["generated_outcome"]["status"] == "published"
    assert fake_keyword_packs.calls == []
