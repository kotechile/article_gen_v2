import sys
from pathlib import Path
import importlib

# Ensure repo root is on sys.path when running tests without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name, supabase):
        self._name = name
        self._supabase = supabase
        self._select_fields = None
        self._filters = []
        self._single = False

    def select(self, fields):
        self._select_fields = fields
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def limit(self, _n):
        return self

    def single(self):
        self._single = True
        return self

    def execute(self):
        if self._name == "llm_providers":
            # Return DB preference first (no key here).
            # The TrendEngine then must resolve api_key from application_settings.
            is_default = any(k == "is_default" and v is True for k, v in self._filters)
            if is_default:
                return _FakeResponse([{"provider": "google", "model_name": "gemini-1.5-flash"}])
            return _FakeResponse([])

        if self._name == "application_settings":
            return _FakeResponse(
                {
                    "geminiKey": "test-gemini-key",
                    "geminiModel": "gemini-1.5-flash",
                    "openAIKey": None,
                    "openAIModel": None,
                    "perplexityAI_key": None,
                    "perplexityModel": None,
                    "claudeKey": None,
                }
            )

        return _FakeResponse([] if not self._single else None)


class _FakeSupabase:
    def table(self, name):
        return _FakeTable(name, self)


def test_trend_engine_requires_llm_api_key(monkeypatch):
    import src.core.supabase_singleton as sb_singleton

    monkeypatch.setattr(sb_singleton, "get_supabase_client", lambda: _FakeSupabase())

    # Reload to pick up the patched get_supabase_client reference during TrendEngine init.
    import src.services.trend_engine as trend_engine
    importlib.reload(trend_engine)

    engine = trend_engine.TrendEngine()
    assert engine.llm.api_key == "test-gemini-key"
    assert engine.llm.default_provider == "google"
