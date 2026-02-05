# Backend API Endpoint Documentation

This document outlines the API endpoints for the backend server and how they are used by the frontend.

## General

*   **`GET /`**
    *   **Description:** Root endpoint with API information.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with a welcome message, API version, status, and a link to the documentation.

*   **`GET /health`**
    *   **Description:** Health check endpoint.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object indicating the API status.

*   **`GET /api/storage/status`**
    *   **Description:** Checks Supabase storage status and configuration.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the status of the Supabase connection.

## Topic Decomposition (OLD FRONTEND)

*   **`POST /api/topic-decomposition`**
    *   **Description:** Decomposes a topic into subtopics using an LLM.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `search_query`, `user_id`, `max_subtopics`, `use_autocomplete`, and `use_llm`.
    *   **Response:** A JSON object containing a list of subtopics.

*   **`POST /api/enhanced-topic-decomposition`**
    *   **Description:** Enhanced topic decomposition with affiliate research and Google Autocomplete simulation.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `search_query`, `user_id`, `max_subtopics`, and `use_llm`.
    *   **Response:** A JSON object containing a list of subtopics.

## Affiliate Research

*   **`GET /api/test-affiliate`**
    *   **Description:** Test endpoint to verify affiliate research routing.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with a success message.

*   **`POST /api/affiliate-research`**
    *   **Description:** Performs affiliate research using a hybrid approach of real affiliate search and topic-specific matching.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `search_term`, `topic`, and `user_id`.
    *   **Response:** A JSON object with a list of affiliate programs.

*   **`POST /api/affiliate-research/search`**
    *   **Description:** Searches for affiliate programs.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/affiliate-research.service.ts` by the `searchAffiliatePrograms` method.
    *   **Request Body:** A JSON object with `search_term`, `niche`, `budget_range`, and `user_id`.
    *   **Response:** A JSON object with a list of affiliate programs.

*   **`POST /api/affiliate-research/content-ideas`**
    *   **Description:** Generates content ideas based on selected affiliate programs.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `selected_programs` and `user_id`.
    *   **Response:** A JSON object with a list of content ideas.

*   **`GET /api/affiliate-research/history/{user_id}`**
    *   **Description:** Gets a user's affiliate research history.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the user's research history.

*   **`GET /api/affiliate-research/programs/{program_id}`**
    *   **Description:** Gets detailed information about a specific affiliate program.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the program details.

*   **`GET /api/affiliate-research/categories`**
    *   **Description:** Gets a list of available affiliate program categories.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/affiliate-research.service.ts` by the `getCategories` method.
    *   **Response:** A JSON object with a list of categories.

*   **`GET /api/affiliate-research/networks`**
    *   **Description:** Gets a list of available affiliate networks.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/affiliate-research.service.ts` by the `getNetworks` method.
    *   **Response:** A JSON object with a list of networks.

## AHREFS Integration

*   **`POST /api/ahrefs/upload`**
    *   **Description:** Uploads and parses an AHREFS CSV file.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A form with a file, `topic_id`, and `user_id`.
    *   **Response:** A JSON object with the processing results.

*   **`POST /api/content-ideas/generate-ahrefs`**
    *   **Description:** Generates content ideas using AHREFS keyword data.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `topic_id`, `topic_title`, `subtopics`, `ahrefs_keywords`, and `user_id`.
    *   **Response:** A JSON object with the generated content ideas.

## Content Idea Generation

*   **`POST /api/content-ideas/generate`**
    *   **Description:** Generates content ideas based on topic, subtopics, and keywords.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/content-ideas.service.ts` by the `generateContentIdeas` method.
    *   **Request Body:** A JSON object with `topic_id`, `topic_title`, `subtopics`, `keywords`, `user_id`, and `content_types`.
    *   **Response:** A JSON object with the generated content ideas.

*   **`POST /api/content-ideas/generate-optimized`**
    *   **Description:** Generates content ideas with optimized keyword loading from the database.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `topic_id`, `topic_title`, `subtopics`, `user_id`, `content_types`, and `max_keywords`.
    *   **Response:** A JSON object with the generated content ideas.

## Keyword Generation

*   **`POST /api/keywords/generate`**
    *   **Description:** Generates simple seed keywords using an LLM.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `subtopics`, `topicId`, and `topicTitle`.
    *   **Response:** A JSON object with a list of keywords.

## Content Idea Management

*   **`POST /api/content-ideas/list`**
    *   **Description:** Retrieves content ideas for a specific topic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/content-ideas.service.ts` by the `getContentIdeas` method.
    *   **Request Body:** A JSON object with `user_id`, `topic_id`, and `content_type`.
    *   **Response:** A list of content ideas.

*   **`POST /api/content-ideas/delete`**
    *   **Description:** Deletes a specific content idea.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `idea_id` and `user_id`.
    *   **Response:** A JSON object with a success message.

*   **`DELETE /api/content-ideas/{idea_id}`**
    *   **Description:** Deletes a specific content idea by ID.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/content-ideas.service.ts` by the `deleteContentIdea` method.
    *   **Response:** A JSON object with a success message.

*   **`POST /api/content-ideas/cleanup`**
    *   **Description:** Deletes all content ideas for a specific topic.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `topic_id` and `user_id`.
    *   **Response:** A JSON object with the cleanup results.

*   **`POST /api/content-ideas/stats`**
    *   **Description:** Gets statistics about content ideas.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `user_id`, `topic_id`.
    *   **Response:** A JSON object with content idea statistics.

*   **`POST /api/content-ideas/publish`**
    *   **Description:** Publishes content ideas to the "Titles" table for the content creation workflow.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/content-ideas.service.ts` by the `publishContentIdeas` method.
    *   **Request Body:** A JSON object with `idea_ids` and `user_id`.
    *   **Response:** A JSON object with a success message.

## DataForSEO

*   **`GET /api/v1/dataforseo/health`**
    *   **Description:** Check DataForSEO API health and connectivity.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the health status.

### Trend Analysis

*   **`POST /api/v1/trend-analysis/dataforseo`**
    *   **Description:** Get trend analysis data for subtopics.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `subtopics`, `location`, and `time_range`.
    *   **Response:** A list of trend data.

*   **`GET /api/v1/trend-analysis/dataforseo` (Functional Router)**
    *   **Description:** Get trend analysis data for subtopics.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Query Parameters:** `subtopics`, `location`, `time_range`, `include_geography`.
    *   **Response:** A list of trend data.

*   **`POST /api/v1/trend-analysis/dataforseo/compare`**
    *   **Description:** Compare trend data for multiple subtopics.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `subtopics`, `location`, and `time_range`.
    *   **Response:** A JSON object with comparison data.

*   **`POST /api/v1/trend-analysis/dataforseo/suggestions`**
    *   **Description:** Get trending subtopic suggestions for a topic.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `topic`, `location`, `time_range`, and `max_suggestions`.
    *   **Response:** A list of trending suggestions.

### Keyword Research

*   **`POST /api/v1/keyword-research/dataforseo`**
    *   **Description:** Research keywords using DataForSEO Labs API.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `seed_keywords`, `max_difficulty`, `max_keywords`, and `location`.
    *   **Response:** A list of researched keywords.

*   **`POST /api/v1/keyword-research/dataforseo` (Functional Router)**
    *   **Description:** Research keywords using DataForSEO Labs API with single keyword input and depth parameter.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `keyword`, `depth`, `max_keywords`, `location_code`, `language_code`.
    *   **Response:** A list of researched keywords.

*   **`POST /api/v1/keyword-research/dataforseo/prioritize`**
    *   **Description:** Prioritize keywords based on various factors.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `keywords`, `priority_factors`.
    *   **Response:** A prioritized list of keywords.

*   **`POST /api/v1/keyword-research/related-keywords`**
    *   **Description:** Get related keywords.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `keywords`, `location_code`, `language_code`, `depth`, and `limit`.
    *   **Response:** A list of related keywords.

*   **`POST /api/v1/keyword-research/dataforseo/related` (Functional Router)**
    *   **Description:** Get related keywords using DataForSEO Labs API with single keyword input.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `keyword`, `depth`, `location_code`, `language_code`, `limit`.
    *   **Response:** A list of related keywords.

*   **`POST /api/v1/keyword-research/keyword-ideas`**
    *   **Description:** Get keyword ideas.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `seed_keywords`, `location_code`, `language_code`, and `limit`.
    *   **Response:** A list of keyword ideas.

*   **`POST /api/v1/keyword-research/store`**
    *   **Description:** Store keyword data in Supabase.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A list of keyword data objects.
    *   **Response:** A JSON object with the success status and count of stored keywords.

*   **`GET /api/v1/keyword-research/test`**
    *   **Description:** Test endpoint for keyword research.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with a success message.

*   **`GET /api/v1/keyword-research/dataforseo/test` (Functional Router)**
    *   **Description:** Test endpoint for DataForSEO keyword research.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with a success message.

*   **`GET /api/v1/keyword-research/debug`**
    *   **Description:** Debug endpoint to check database structure and data.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with sample data and column information.

*   **`GET /api/v1/keyword-research/by-topic/{topic_id}`**
    *   **Description:** Get all keywords for a specific topic and user.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A list of keyword data.

## Article Generation

*   **`POST /api/article/generate`**
    *   **Description:** Generates an article from a brief, with optional RAG integration.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with article generation parameters.
    *   **Response:** A JSON object with the generated article and metadata.

*   **`POST /api/research/generate`**
    *   **Description:** An alias for `/api/article/generate` that logs the payload.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with article generation parameters.
    *   **Response:** A JSON object with the generated article and metadata.

## Research Topics (NEW FRONTEND)

*   **`POST /api/research-topics/`**
    *   **Description:** Creates a new research topic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `createResearchTopic` method.
    *   **Request Body:** A JSON object with the research topic data.
    *   **Response:** The created research topic.

*   **`GET /api/research-topics/{topic_id}`**
    *   **Description:** Gets a research topic by ID.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `getResearchTopic` method.
    *   **Response:** The research topic data.

*   **`GET /api/research-topics/`**
    *   **Description:** Lists research topics with pagination and filtering.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `listResearchTopics` method.
    *   **Response:** A list of research topics.

*   **`PUT /api/research-topics/{topic_id}`**
    *   **Description:** Updates a research topic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `updateResearchTopic` method.
    *   **Request Body:** A JSON object with the updated research topic data.
    *   **Response:** The updated research topic.

*   **`DELETE /api/research-topics/{topic_id}`**
    *   **Description:** Deletes a research topic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `deleteResearchTopic` method.
    *   **Response:** A 204 No Content response.

*   **`GET /api/research-topics/{topic_id}/stats`**
    *   **Description:** Gets statistics for a research topic.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with research topic statistics.

*   **`GET /api/research-topics/stats/overview`**
    *   **Description:** Gets overview statistics for all research topics.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/research-topics.service.ts` by the `getOverviewStats` method.
    *   **Response:** A JSON object with overview statistics.

### Subtopics

*   **`POST /api/research-topics/{topic_id}/subtopics`**
    *   **Description:** Creates a new subtopic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `createSubtopic` method.
    *   **Request Body:** A JSON object with the subtopic data.
    *   **Response:** The created subtopic.

*   **`GET /api/research-topics/{topic_id}/subtopics`**
    *   **Description:** Gets enriched subtopics for a topic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `getSubtopics` method.
    *   **Response:** A list of subtopics.

*   **`POST /api/research-topics/{topic_id}/subtopics/generate`**
    *   **Description:** Generates subtopics using an LLM and saves them.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `generateSubtopics` method.
    *   **Response:** A list of generated subtopics.

*   **`POST /api/research-topics/{topic_id}/enrich`**
    *   **Description:** Enriches subtopics with real-world data (trends, affiliate opportunities).
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `enrichSubtopics` method.
    *   **Response:** A list of enriched subtopics.

*   **`PUT /api/research-topics/{topic_id}/subtopics/{subtopic_id}`**
    *   **Description:** Updates a subtopic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `updateSubtopic` method.
    *   **Request Body:** A JSON object with the updated subtopic data.
    *   **Response:** The updated subtopic.

*   **`DELETE /api/research-topics/{topic_id}/subtopics/{subtopic_id}`**
    *   **Description:** Deletes a subtopic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/subtopics.service.ts` by the `deleteSubtopic` method.
    *   **Response:** A 204 No Content response.

## Keyword Management

*   **`POST /api/keywords/upload`**
    *   **Description:** Uploads keywords from a CSV file.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the upload results.

*   **`POST /api/keywords/crawl`**
    *   **Description:** Crawls keywords using DataForSEO.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Request Body:** A JSON object with `seed_keyword`, `depth`, `geo`, and `language`.
    *   **Response:** A JSON object with the crawl results.

*   **`GET /api/keywords/data/{keyword_data_id}`**
    *   **Description:** Gets keyword data by ID.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the keyword data.

*   **`GET /api/keywords/data`**
    *   **Description:** Lists a user's keyword data.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A list of keyword data.

*   **`DELETE /api/keywords/data/{keyword_data_id}`**
    *   **Description:** Deletes keyword data.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A success message.

*   **`GET /api/keywords/data/{keyword_data_id}/analysis`**
    *   **Description:** Gets keyword analysis.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the keyword analysis.

*   **`GET /api/keywords/data/{keyword_data_id}/clusters`**
    *   **Description:** Gets keyword clusters.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A list of keyword clusters.

*   **`POST /api/keywords/data/{keyword_data_id}/enrich`**
    *   **Description:** Enriches keywords with additional data.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A success message.

*   **`POST /api/keywords/data/{keyword_data_id}/cluster`**
    *   **Description:** Clusters keywords.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A success message.

*   **`GET /api/keywords/data/{keyword_data_id}/export`**
    *   **Description:** Exports keywords to a file.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with the download URL.

*   **`GET /api/keywords/suggestions`**
    *   **Description:** Gets keyword suggestions.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with keyword suggestions.

*   **`GET /api/keywords/data/{keyword_data_id}/analytics`**
    *   **Description:** Gets keyword analytics.
    *   **Frontend Usage:** Not used directly by the frontend.
    *   **Response:** A JSON object with keyword analytics.

## Enhanced Topics

*   **`POST /api/enhanced-topics/idea-burst`**
    *   **Description:** Generates blog titles and software ideas for a specific subtopic.
    *   **Frontend Usage:** Used in `profit-path-frontend/lib/services/content-ideas.service.ts` by the `generateBurst` method.
    *   **Request Body:** A JSON object with `user_id`, `subtopic`, `keywords`, and `affiliate_offers`.
    *   **Response:** A JSON object with the generated ideas.
