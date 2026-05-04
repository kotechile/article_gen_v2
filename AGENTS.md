# AGENTS.md

## Purpose

This file is the working guide for agents editing this repository. It reflects the current frontend behavior, the WordPress handoff path, and the current limitations around rebuilding imported WordPress articles inside an Astro app.

## Repo Areas That Matter Most

- `frontend/`: main React/Vite app used for research, content ideas, Content Studio, and WordPress export.
- `src/api/endpoints/research_topics.py`: topic and idea-burst orchestration, including category-aware content idea persistence.
- `src/api/endpoints/content_ideas.py`: publishing content ideas into `Titles`.
- `src/api/wordpress.py`: WordPress post/category sync endpoints.
- `src/api/internal_links.py`: internal-link suggestions based on imported WordPress posts.

## Current Frontend State

### Content Studio

File: [frontend/src/pages/ContentStudio.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/pages/ContentStudio.tsx)

Recent behavior that must be preserved:

- Content Studio now resolves and displays the full category path, not just a single WordPress category ID.
- Category path resolution should prefer `Titles.idea_metadata.category_context.category_path`.
- If that is missing, it should fall back to:
  1. `Titles.topic_id`
  2. `Titles.source_idea_id -> content_ideas.topic_id`
  3. `research_topics.primary_category_id + secondary_category_id -> project_categories.name`
- `Titles` records created from ideas should carry `topic_id` and `idea_metadata` so downstream UI does not lose category/subcategory context.

Important implication:

- If an agent touches article loading in Content Studio, do not regress the `topic_id` and `source_idea_id` fallback chain.
- If an agent changes idea publishing or article persistence, verify that Level1 and Level2 category context still appears in Content Studio.

### WordPress Export Modal

File: [frontend/src/components/WordPressExportModal.tsx](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/components/WordPressExportModal.tsx)

Recent behavior that must be preserved:

- Category auto-selection is no longer single-source.
- The modal now merges category candidates from:
  1. `articleData.wordpress_category_id`
  2. previously saved export settings
  3. linked topic/project category mappings via `resolveLinkedWordPressCategoryIds`
- This is specifically to preserve Level2/subcategory handoff to WordPress.
- When the linked resolver returns both subcategory and primary category, both should remain selected.

Important implication:

- Do not reintroduce early returns after the first matching category.
- When editing WordPress export UX, remember that multiple category IDs are intentional.

### WordPress Service Expectations

File: [frontend/src/services/wordpressService.ts](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/frontend/src/services/wordpressService.ts)

Expected behavior:

- `publishToWordPress(...)` sends `categories: number[]` to WordPress, not a single category.
- `resolveLinkedWordPressCategoryIds(...)` should prefer subcategory first, then primary category.
- Publishing loopback updates `Titles` with post status, post URL, post ID, and canonicalized SEO title/description.

## Backend Persistence Rules

### Idea Burst Persistence

File: [src/api/endpoints/research_topics.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/research_topics.py)

Current expectations:

- New content ideas should persist `idea_metadata.category_context`.
- `category_context` should include:
  - `project_id`
  - `primary_category_id`
  - `secondary_category_id`
  - `primary_category_name`
  - `secondary_category_name`
  - `primary_category_description`
  - `secondary_category_description`
  - `category_path`
- This context exists to keep category/subcategory information available after the user leaves the Research view.

If an agent edits topic generation or idea persistence:

- Keep category/subcategory traceability intact.
- Do not collapse the context down to a single generic `category` string.

### Publishing Ideas Into Titles

File: [src/api/endpoints/content_ideas.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/endpoints/content_ideas.py)

Current expectations:

- When a blog idea is published into `Titles`, the inserted record should include:
  - `source_idea_id`
  - `topic_id`
  - `idea_metadata`
  - WordPress mapping fields
  - keyword handoff fields
- This is required so Content Studio and WordPress export can still derive category path and topic lineage later.

If an agent edits this path:

- Verify that Titles created from content ideas still preserve topic linkage.
- Verify that category/subcategory context still reaches Content Studio and WordPress export.

## Category/Subcategory Contract

The current product expectation is:

- topic generation starts from project category/subcategory
- content ideas retain both Level1 and Level2 context
- Content Studio can recover and display that context
- WordPress export can send both matching categories when available

In practical terms:

- `primary_category_id` is Level1
- `secondary_category_id` is Level2
- `category_path` should usually be `Level1 / Level2`

If an agent sees only one generic `category` field being used, that is not enough for the current workflow.

## WordPress Import Reality

File: [src/api/wordpress.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/wordpress.py)

Current import behavior:

- `POST /api/wordpress/sync-posts` fetches recent posts from configured WordPress sites.
- Imported rows saved to `wordpress_imported_posts` currently contain only:
  - `user_id`
  - `wordpress_detail_id`
  - `post_id`
  - `title`
  - `link`
  - `excerpt`

What this means:

- The current sync is good enough for internal-link suggestions.
- It is not sufficient to recreate full articles inside an Astro app.
- There is no Astro app in this repository today, so any Astro rebuild flow is either external or still to be created.

## Recreating WordPress Articles In Astro

If the goal is to read WordPress articles and recreate them in an Astro app, agents should treat the current system as incomplete and extend it deliberately.

### Minimum Data Needed

To reconstruct articles in Astro reliably, imported records should also store:

- `slug`
- `content.rendered` or cleaned article HTML
- `date`
- `modified`
- `status`
- `author`
- `featured_media`
- resolved media URL
- category IDs and category names
- tag IDs and tag names
- SEO metadata if available
  - Yoast/RankMath title
  - meta description
  - canonical URL
- source domain

### Recommended Import Upgrade

When extending `sync-posts`, prefer storing richer post snapshots in `wordpress_imported_posts` or a new dedicated table such as `wordpress_imported_articles`.

Recommended snapshot fields:

- `post_id`
- `slug`
- `title`
- `excerpt`
- `content_html`
- `link`
- `published_at`
- `modified_at`
- `featured_image_url`
- `category_ids`
- `category_names`
- `tag_ids`
- `tag_names`
- `seo_title`
- `seo_description`
- `canonical_url`
- `raw_post_json`

### Astro Reconstruction Guidance

For an Astro app, agents should prefer this pipeline:

1. Sync full WordPress article data into Supabase.
2. Normalize HTML enough for Astro rendering.
3. Preserve canonical metadata and taxonomy.
4. Convert or wrap content into Astro-compatible templates.
5. Rebuild internal links using imported post inventory.

Important caveat:

- Do not promise lossless Astro recreation from the current `wordpress_imported_posts` table. Right now it only stores title, link, and excerpt.

## Internal Linking

File: [src/api/internal_links.py](/Users/jorgefernandezilufi/Documents/_article_research/content_generator/content_generator_v2/src/api/internal_links.py)

Current behavior:

- Internal link suggestions are generated from `wordpress_imported_posts`.
- Because imported post data is currently shallow, suggestions are title/link-driven rather than full-content-aware.

If the WordPress import is upgraded for Astro reconstruction:

- Consider upgrading internal-link candidate quality to use article body, categories, tags, and slug context.

## Agent Checklist Before Shipping Changes

- If you touch Content Studio, verify category path still appears for Titles created from ideas.
- If you touch WordPress export, verify multiple category IDs can still be sent.
- If you touch idea persistence, verify `topic_id`, `source_idea_id`, and `idea_metadata.category_context` survive.
- If you touch WordPress sync, be explicit whether you are optimizing for internal-link suggestions or full Astro article reconstruction.
- If you add Astro-related work, document whether the Astro app lives inside this repo or is an external consumer.

## Known Limitation To Keep In Mind

The system currently has two different WordPress-related use cases:

- publish newly generated content to WordPress
- read existing WordPress content back for reuse

These are not symmetric today. Publishing is much richer than importing. Agents should not assume the import side already has enough data to recreate a post in Astro without schema and endpoint upgrades.
