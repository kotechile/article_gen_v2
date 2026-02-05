# Database Usage Documentation

This document outlines the database tables and how they are used by the backend, and which tables and fields are not used by the frontend.

## Tables and Fields Used by the Frontend

This section describes the tables and fields that are directly or indirectly used by the frontend application.

### `research_topics`

*   **Purpose:** Stores the main research topics that users create.
*   **Backend Interaction:** The `research_topic_service` in `backend/src/services/research_topic_service.py` is responsible for all interactions with this table.
*   **Frontend Usage:** All fields in this table are used by the frontend.

| Column | Description |
| :--- | :--- |
| `id` | Unique identifier for the research topic. |
| `user_id` | Foreign key to the `users` table, indicating who owns the topic. |
| `title` | The title of the research topic. |
| `description` | A detailed description of the research topic. |
| `status` | The status of the research topic (e.g., `active`, `completed`, `archived`). |
| `created_at` | Timestamp of when the topic was created. |
| `updated_at` | Timestamp of when the topic was last updated. |
| `version` | Version number for optimistic concurrency control. |

### `subtopics`

*   **Purpose:** Stores the subtopics that are generated for a research topic.
*   **Backend Interaction:** The `subtopics_service` in `backend/src/services/subtopics_service.py` is responsible for all interactions with this table.
*   **Frontend Usage:** All fields in this table are used by the frontend.

| Column | Description |
| :--- | :--- |
| `id` | Unique identifier for the subtopic. |
| `research_topic_id` | Foreign key to the `research_topics` table. |
| `user_id` | Foreign key to the `users` table. |
| `name` | The name of the subtopic. |
| `trend_direction` | The direction of the trend for the subtopic (e.g., `up`, `down`, `stable`). |
| `trend_score` | A score from 0-100 indicating the strength of the trend. |
| `interest_over_time` | A JSONB array of data points showing interest over time. |
| `seo_difficulty` | A score from 0-100 indicating the SEO difficulty of the subtopic. |
| `search_volume` | The monthly search volume for the subtopic. |
| `cpc` | The cost-per-click for the subtopic. |
| `affiliate_offer_count` | The number of affiliate offers available for the subtopic. |
| `keywords` | A JSONB array of keywords related to the subtopic. |
| `viability_score` | A calculated score from 0-100 indicating the overall viability of the subtopic. |
| `created_at` | Timestamp of when the subtopic was created. |
| `updated_at` | Timestamp of when the subtopic was last updated. |
| `rationale` | The rationale behind the generation of the subtopic. |
| `target_audience` | The target audience for the subtopic. |

### `content_ideas`

*   **Purpose:** Stores content ideas for a specific topic.
*   **Backend Interaction:** The `content_idea_service` in `backend/src/services/content_idea_service.py` is responsible for interactions with this table.
*   **Frontend Usage:** The following fields are used by the frontend: `id`, `title`, `content_type`, `primary_keywords`, `secondary_keywords`, `seo_optimization_score`, `traffic_potential_score`, `total_search_volume`, `average_difficulty`, `average_cpc`, `created_at`, `updated_at`, `status`, `user_id`, `topic_id`, `subtopic`, `description`, `published`, `published_at`, `published_to_titles`, `titles_record_id`, `viability_score`, `trend_score`, `monetization_score`, `seo_ease_score`.

### `affiliate_programs`

*   **Purpose:** Stores information about affiliate programs.
*   **Backend Interaction:** The `affiliate_research_service` in `backend/src/services/affiliate_research_service.py` is responsible for interactions with this table. The frontend does not interact with this table directly, but it is used by the `/api/affiliate-research/search` endpoint.
*   **Frontend Usage:** The frontend uses a simplified `AffiliateProgram` interface. The fields used are: `id`, `name`, `commission`, `url`, `description`. The `trustScore` field is a frontend-only field.

## Tables and Fields Not Used by the Frontend

This section describes the tables and fields that are not used directly or indirectly by the frontend application.

### Unused Tables

The following tables are not used by the frontend:

*   `PlannedArticles`
*   `PostLinks`
*   `RSS`
*   `TableOfContents`
*   `Titles`
*   `Titles_citations`
*   `Tones`
*   `affiliate_offers`
*   `affiliate_research`
*   `api_keys`
*   `application_settings`
*   `blog_generation_results`
*   `blog_idea_keyword_assignments`
*   `blog_idea_performance`
*   `blog_idea_templates`
*   `blog_ideas`
*   `categoriesByPost`
*   `competitive_intelligence`
*   `content_calendar`
*   `content_opportunities`
*   `dataforseo_api_logs`
*   `embeddings`
*   `geographic_insights`
*   `imported_keywords`
*   `indexed_documents`
*   `infographic`
*   `infographicDetails`
*   `keyword_intelligence`
*   `keyword_opportunities_reports`
*   `keyword_research_data`
*   `keyword_research_sessions`
*   `keywords`
*   `lindex_collections`
*   `lindex_documents`
*   `lindex_embedding_chunk`
*   `lindex_sections`
*   `llm_configurations`
*   `llm_providers`
*   `manual_action_suggestions`
*   `mySources`
*   `offer_analytics`
*   `offer_research_sessions`
*   `postTypes`
*   `research_program_links`
*   `seasonal_calendar`
*   `sectionSpecificPrompts`
*   `subtopic_suggestions`
*   `summaries`
*   `topic_decompositions`
*   `trend_analyses`
*   `trend_analysis`
*   `trend_analysis_data`
*   `trend_predictions`
*   `trending_topics`
*   `user_offer_preferences`
*   `user_profile`
*   `users`
*   `wordPress_details`
*   `vecs.RAG_HOUSE_AND_REAL_ESTATE_128D_OPT`
*   `vecs.default`
*   `vecs.general_knowledge`
*   `vecs.house`
*   `vecs.house_and_real_estate`
*   `vecs.rag_house`
*   `vecs.rag_house_and_real_estate`
*   `vecs.rag_house_and_real_estate_128D`
*   `vecs.rag_house_and_real_estate_128D_OPT`
*   `vecs.rag_house_and_real_estate_128d_opt`
*   `vecs.real_estate`
*   `vecs.real_estate_128D_OPT`
*   `vecs.research_documents`
*   `vecs.test`
*   `vecs.test-collection`
*   `vecs.your_collection`

### Unused Fields in Used Tables

The following fields in tables that are used by the frontend are not used by the frontend application.

#### `content_ideas`

*   `workflow_session_id`
*   `trend_analysis_id`
*   `topic_decomposition_id`
*   `priority`
*   `key_points`
*   `content_outline`
*   `primary_keyword`
*   `affiliate_offers`
*   `affiliate_links`
*   `monetization_strategy`
*   `expected_revenue`
*   `generation_prompt`
*   `generation_model`
*   `generation_parameters`
*   `generation_time_ms`
*   `readability_score`
*   `seo_score`
*   `engagement_score`
*   `quality_notes`
*   `target_publish_date`
*   `actual_publish_date`
*   `publish_url`
*   `word_count`
*   `reading_time_minutes`
*   `tags`
*   `categories`
*   `workflow_status`
*   `content_generated`
*   `content_brief_generated`
*   `viral_potential_score`
*   `competition_score`
*   `keyword_research_data`
*   `keyword_research_enhanced`
*   `affiliate_opportunities`
*   `estimated_annual_revenue`
*   `monetization_priority`
*   `generation_method`
*   `enhancement_timestamp`
*   `estimated_word_count`

#### `affiliate_programs`

*   `company_name`
*   `website_url`
*   `network_name`
*   `commission_type`
*   `cookie_duration`
*   `payment_terms`
*   `application_requirements`
*   `program_url`
*   `contact_email`
*   `status`
*   `verification_status`
*   `last_verified`
*   `research_score`
*   `popularity_score`
*   `conversion_rate`
*   `avg_order_value`
*   `target_audience`
*   `content_opportunities`
*   `seasonal_trends`
*   `competitor_analysis`
*   `source`
*   `data_quality_score`
*   `last_researched`
*   `research_count`