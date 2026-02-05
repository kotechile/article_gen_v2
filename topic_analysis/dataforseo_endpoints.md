## DataForSEO Endpoints and their API Calls

This section lists the backend endpoints that interact with the DataForSEO API and details the specific DataForSEO API endpoints they call.

### `backend/src/routers/dataforseo_router.py`

*   **`POST /api/v1/trend-analysis/dataforseo`**
    *   **Description:** Get trend analysis data for specified subtopics.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live` (or similar Google Trends endpoint via `api_client.get_trend_data`)

*   **`POST /api/v1/trend-analysis/dataforseo/compare`**
    *   **Description:** Compare multiple subtopics' trend data.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live` (or similar Google Trends endpoint via `api_client.get_trend_data`)

*   **`POST /api/v1/trend-analysis/dataforseo/suggestions`**
    *   **Description:** Get trending subtopic suggestions for a given topic.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/related_queries/live` or `https://api.dataforseo.com/v3/keywords_data/google_trends/all/live` (or similar via `api_client.get_suggestions`)

*   **`POST /api/v1/keyword-research/dataforseo`**
    *   **Description:** Research keywords using a 2-call approach (keyword ideas and related keywords).
    *   **DataForSEO API Endpoints:**
        *   `https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live` (via `dataforseo_service.get_keyword_ideas`)
        *   `https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live` (via `dataforseo_service.get_related_keywords`)

*   **`POST /api/v1/keyword-research/dataforseo/prioritize`**
    *   **Description:** Prioritize keywords based on various factors.
    *   **DataForSEO API Endpoint:** No direct external API call; processes previously fetched DataForSEO data.

*   **`GET /api/v1/dataforseo/health`**
    *   **Description:** Check DataForSEO API health and connectivity.
    *   **DataForSEO API Endpoint:** A minimal DataForSEO endpoint call (e.g., `info`, `ping`, or a minimal trends/keywords request) is made internally by `DataForSEOAPIClient().initialize()`.

*   **`POST /api/v1/keyword-research/related-keywords`**
    *   **Description:** Get related keywords using a single-step live endpoint.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live` (via `dataforseo_service.get_related_keywords`)

*   **`POST /api/v1/keyword-research/keyword-ideas`**
    *   **Description:** Get keyword ideas.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live` (via `dataforseo_service.get_keyword_ideas`)

### `backend/src/routers/functional_dataforseo_router.py`

*   **`GET /api/v1/dataforseo/health`**
    *   **Description:** Check DataForSEO API health and connectivity.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live` (used for a test call via `make_dataforseo_request`)

*   **`GET /api/v1/trend-analysis/dataforseo`**
    *   **Description:** Get trend analysis data for specified subtopics.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live` (via `make_dataforseo_request`)

*   **`POST /api/v1/keyword-research/dataforseo`**
    *   **Description:** Research keywords using single keyword input and depth parameter.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live` (via `make_dataforseo_request`)

*   **`POST /api/v1/keyword-research/dataforseo/related`**
    *   **Description:** Get related keywords using single keyword input.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live` (via `make_dataforseo_request`)

*   **`POST /api/v1/trend-analysis/dataforseo/compare`**
    *   **Description:** Compare multiple subtopics' trend data.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live` (called for each subtopic via `make_dataforseo_request`)

*   **`POST /api/v1/trend-analysis/dataforseo/suggestions`**
    *   **Description:** Get trending subtopic suggestions for a given topic.
    *   **DataForSEO API Endpoint:** `https://api.dataforseo.com/v3/keywords_data/google_ads/keywords_for_keywords/live` (via `make_dataforseo_request`)
