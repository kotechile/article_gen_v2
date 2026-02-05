create extension if not exists "vector" with schema "extensions";


create sequence "public"."documents_id_seq";

create sequence "public"."indexed_documents_id_seq";

revoke delete on table "public"."api_keys" from "anon";

revoke insert on table "public"."api_keys" from "anon";

revoke references on table "public"."api_keys" from "anon";

revoke select on table "public"."api_keys" from "anon";

revoke trigger on table "public"."api_keys" from "anon";

revoke truncate on table "public"."api_keys" from "anon";

revoke update on table "public"."api_keys" from "anon";

revoke delete on table "public"."api_keys" from "authenticated";

revoke insert on table "public"."api_keys" from "authenticated";

revoke references on table "public"."api_keys" from "authenticated";

revoke select on table "public"."api_keys" from "authenticated";

revoke trigger on table "public"."api_keys" from "authenticated";

revoke truncate on table "public"."api_keys" from "authenticated";

revoke update on table "public"."api_keys" from "authenticated";

revoke delete on table "public"."api_keys" from "service_role";

revoke insert on table "public"."api_keys" from "service_role";

revoke references on table "public"."api_keys" from "service_role";

revoke select on table "public"."api_keys" from "service_role";

revoke trigger on table "public"."api_keys" from "service_role";

revoke truncate on table "public"."api_keys" from "service_role";

revoke update on table "public"."api_keys" from "service_role";

revoke delete on table "public"."dataforseo_api_logs" from "anon";

revoke insert on table "public"."dataforseo_api_logs" from "anon";

revoke references on table "public"."dataforseo_api_logs" from "anon";

revoke select on table "public"."dataforseo_api_logs" from "anon";

revoke trigger on table "public"."dataforseo_api_logs" from "anon";

revoke truncate on table "public"."dataforseo_api_logs" from "anon";

revoke update on table "public"."dataforseo_api_logs" from "anon";

revoke delete on table "public"."dataforseo_api_logs" from "authenticated";

revoke insert on table "public"."dataforseo_api_logs" from "authenticated";

revoke references on table "public"."dataforseo_api_logs" from "authenticated";

revoke select on table "public"."dataforseo_api_logs" from "authenticated";

revoke trigger on table "public"."dataforseo_api_logs" from "authenticated";

revoke truncate on table "public"."dataforseo_api_logs" from "authenticated";

revoke update on table "public"."dataforseo_api_logs" from "authenticated";

revoke delete on table "public"."dataforseo_api_logs" from "service_role";

revoke insert on table "public"."dataforseo_api_logs" from "service_role";

revoke references on table "public"."dataforseo_api_logs" from "service_role";

revoke select on table "public"."dataforseo_api_logs" from "service_role";

revoke trigger on table "public"."dataforseo_api_logs" from "service_role";

revoke truncate on table "public"."dataforseo_api_logs" from "service_role";

revoke update on table "public"."dataforseo_api_logs" from "service_role";

revoke delete on table "public"."keyword_research_data" from "anon";

revoke insert on table "public"."keyword_research_data" from "anon";

revoke references on table "public"."keyword_research_data" from "anon";

revoke select on table "public"."keyword_research_data" from "anon";

revoke trigger on table "public"."keyword_research_data" from "anon";

revoke truncate on table "public"."keyword_research_data" from "anon";

revoke update on table "public"."keyword_research_data" from "anon";

revoke delete on table "public"."keyword_research_data" from "authenticated";

revoke insert on table "public"."keyword_research_data" from "authenticated";

revoke references on table "public"."keyword_research_data" from "authenticated";

revoke select on table "public"."keyword_research_data" from "authenticated";

revoke trigger on table "public"."keyword_research_data" from "authenticated";

revoke truncate on table "public"."keyword_research_data" from "authenticated";

revoke update on table "public"."keyword_research_data" from "authenticated";

revoke delete on table "public"."keyword_research_data" from "service_role";

revoke insert on table "public"."keyword_research_data" from "service_role";

revoke references on table "public"."keyword_research_data" from "service_role";

revoke select on table "public"."keyword_research_data" from "service_role";

revoke trigger on table "public"."keyword_research_data" from "service_role";

revoke truncate on table "public"."keyword_research_data" from "service_role";

revoke update on table "public"."keyword_research_data" from "service_role";

revoke delete on table "public"."subtopic_suggestions" from "anon";

revoke insert on table "public"."subtopic_suggestions" from "anon";

revoke references on table "public"."subtopic_suggestions" from "anon";

revoke select on table "public"."subtopic_suggestions" from "anon";

revoke trigger on table "public"."subtopic_suggestions" from "anon";

revoke truncate on table "public"."subtopic_suggestions" from "anon";

revoke update on table "public"."subtopic_suggestions" from "anon";

revoke delete on table "public"."subtopic_suggestions" from "authenticated";

revoke insert on table "public"."subtopic_suggestions" from "authenticated";

revoke references on table "public"."subtopic_suggestions" from "authenticated";

revoke select on table "public"."subtopic_suggestions" from "authenticated";

revoke trigger on table "public"."subtopic_suggestions" from "authenticated";

revoke truncate on table "public"."subtopic_suggestions" from "authenticated";

revoke update on table "public"."subtopic_suggestions" from "authenticated";

revoke delete on table "public"."subtopic_suggestions" from "service_role";

revoke insert on table "public"."subtopic_suggestions" from "service_role";

revoke references on table "public"."subtopic_suggestions" from "service_role";

revoke select on table "public"."subtopic_suggestions" from "service_role";

revoke trigger on table "public"."subtopic_suggestions" from "service_role";

revoke truncate on table "public"."subtopic_suggestions" from "service_role";

revoke update on table "public"."subtopic_suggestions" from "service_role";

revoke delete on table "public"."trend_analysis_data" from "anon";

revoke insert on table "public"."trend_analysis_data" from "anon";

revoke references on table "public"."trend_analysis_data" from "anon";

revoke select on table "public"."trend_analysis_data" from "anon";

revoke trigger on table "public"."trend_analysis_data" from "anon";

revoke truncate on table "public"."trend_analysis_data" from "anon";

revoke update on table "public"."trend_analysis_data" from "anon";

revoke delete on table "public"."trend_analysis_data" from "authenticated";

revoke insert on table "public"."trend_analysis_data" from "authenticated";

revoke references on table "public"."trend_analysis_data" from "authenticated";

revoke select on table "public"."trend_analysis_data" from "authenticated";

revoke trigger on table "public"."trend_analysis_data" from "authenticated";

revoke truncate on table "public"."trend_analysis_data" from "authenticated";

revoke update on table "public"."trend_analysis_data" from "authenticated";

revoke delete on table "public"."trend_analysis_data" from "service_role";

revoke insert on table "public"."trend_analysis_data" from "service_role";

revoke references on table "public"."trend_analysis_data" from "service_role";

revoke select on table "public"."trend_analysis_data" from "service_role";

revoke trigger on table "public"."trend_analysis_data" from "service_role";

revoke truncate on table "public"."trend_analysis_data" from "service_role";

revoke update on table "public"."trend_analysis_data" from "service_role";

alter table "public"."api_keys" drop constraint "api_keys_base_url_check";

alter table "public"."api_keys" drop constraint "api_keys_key_value_check";

alter table "public"."api_keys" drop constraint "api_keys_provider_check";

create table "public"."PlannedArticles" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "user_id" uuid,
    "title" text,
    "link" text,
    "content" text,
    "rssId" text,
    "keyWords" text
);


alter table "public"."PlannedArticles" enable row level security;

create table "public"."PostLinks" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "titleId" text,
    "user_id" uuid default gen_random_uuid(),
    "textKeyWord" text,
    "linkKeyword" text,
    "linkUrl" text default ''::text
);


alter table "public"."PostLinks" enable row level security;

create table "public"."RSS" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "user_id" uuid,
    "url" text,
    "title" text,
    "favorites" boolean
);


alter table "public"."RSS" enable row level security;

create table "public"."TableOfContents" (
    "TitleID" text,
    "chapter" text,
    "selected" boolean,
    "headerType" text,
    "sequence" real,
    "content" text,
    "ImageURL" text,
    "imageFileName" text,
    "user_id" uuid,
    "id" uuid not null default gen_random_uuid(),
    "relevantWords" text,
    "youTubeUrl" text,
    "mediaTitle" text,
    "mediaCaption" text,
    "mediaDescription" text,
    "mediaAltText" text,
    "mediaAuthor" text,
    "contentOutline" text,
    "affiliateLinkId" text,
    "affiliateImageUrl" text,
    "affiliateLinkUrl" text,
    "lengthInWords" text,
    "inputSummaries" text,
    "imageLink" text,
    "hasSubSections" boolean default false,
    "tableInstructions" text,
    "recordType" text,
    "level" text,
    "Level" text,
    "citation_id" text
);


alter table "public"."TableOfContents" enable row level security;

create table "public"."Titles" (
    "Tone" text,
    "Title" text,
    "userDescription" text,
    "articleLength" text,
    "Keywords" text,
    "tocString" text,
    "user_id" uuid,
    "id" uuid not null default gen_random_uuid(),
    "Wordpress_post_Id" text,
    "hook" text,
    "thesis" text,
    "imageFileName" text,
    "ImageURL" text,
    "ImageAuthor" text,
    "dateCreatedOn" timestamp with time zone,
    "mediaAltText" text,
    "mediaTitle" text,
    "mediaCaption" text,
    "metaTitle" text,
    "metaDescription" text,
    "published" boolean,
    "mediaDescription" text,
    "datePublished" date,
    "domain" text,
    "sidebar_id" text,
    "excerpt" text,
    "postType" text,
    "tableOfContentsFlag" boolean default true,
    "sectionNumberingFlag" boolean default true,
    "autoGenerateFeaturedImage" boolean default false,
    "postFutureDate" date,
    "postStatusParam" text,
    "LLM" text,
    "categories" text,
    "affiliateDisclosure" boolean,
    "iterativeGeneration" boolean default false,
    "articleText" text,
    "htmlArticle" text,
    "blog_idea_id" uuid,
    "trend_analysis_id" uuid,
    "content_format" text default 'how_to_guide'::text,
    "difficulty_level" text default 'intermediate'::text,
    "estimated_word_count" integer default 2500,
    "estimated_reading_time" integer default 10,
    "target_audience" text default 'professional'::text,
    "overall_quality_score" integer default 0,
    "viral_potential_score" integer default 0,
    "seo_optimization_score" integer default 0,
    "audience_alignment_score" integer default 0,
    "content_feasibility_score" integer default 0,
    "business_impact_score" integer default 0,
    "primary_keywords_json" jsonb default '[]'::jsonb,
    "secondary_keywords_json" jsonb default '[]'::jsonb,
    "enhanced_primary_keywords" jsonb default '[]'::jsonb,
    "enhanced_secondary_keywords" jsonb default '[]'::jsonb,
    "keyword_research_data" jsonb default '{}'::jsonb,
    "total_search_volume" integer default 0,
    "avg_keyword_difficulty" numeric(5,2) default 0,
    "traffic_potential_score" integer default 0,
    "competition_score" integer default 0,
    "featured_snippet_opportunity" boolean default false,
    "content_outline" jsonb default '[]'::jsonb,
    "key_points" jsonb default '[]'::jsonb,
    "engagement_hooks" jsonb default '[]'::jsonb,
    "visual_elements" jsonb default '[]'::jsonb,
    "call_to_action_text" text,
    "business_value" text,
    "wp_category_ids" jsonb default '[]'::jsonb,
    "wp_tag_ids" jsonb default '[]'::jsonb,
    "wp_custom_fields" jsonb default '{}'::jsonb,
    "wp_featured_image_url" text,
    "wp_excerpt_auto_generated" text,
    "wp_slug" text,
    "wp_status" text default 'draft'::text,
    "focus_keyword" text,
    "seo_title_optimized" text,
    "seo_meta_desc_optimized" text,
    "canonical_url" text,
    "robots_meta" text default 'index,follow'::text,
    "schema_type" text default 'Article'::text,
    "breadcrumb_title" text,
    "readability_score" numeric(5,2) default 0,
    "keyword_density" numeric(5,2) default 0,
    "internal_links_suggested" jsonb default '[]'::jsonb,
    "external_links_suggested" jsonb default '[]'::jsonb,
    "content_optimization_tips" jsonb default '[]'::jsonb,
    "priority_level" text default 'medium'::text,
    "scheduled_publish_date" timestamp with time zone,
    "content_series" text,
    "pillar_post_id" uuid,
    "content_cluster" text,
    "target_keywords_for_tracking" jsonb default '[]'::jsonb,
    "expected_monthly_traffic" integer default 0,
    "expected_conversion_rate" numeric(5,2) default 0,
    "business_goal" text,
    "generation_source" text default 'trend_analysis'::text,
    "source_topic_id" uuid,
    "source_opportunity_id" uuid,
    "keyword_research_enhanced" boolean default false,
    "enhancement_timestamp" timestamp with time zone,
    "workflow_status" text default 'idea_selected'::text,
    "assigned_writer" text,
    "writer_notes" text,
    "editor_notes" text,
    "content_brief_generated" boolean default false,
    "last_updated" timestamp with time zone default now(),
    "updated_by" uuid,
    "content_generated" boolean,
    "status" text,
    "knowledge_gaps_closed" boolean default false,
    "gap_closure_timestamp" timestamp with time zone,
    "gap_closure_method" text,
    "gap_closure_details" jsonb,
    "knowledge_enhanced" boolean default false,
    "rag_collection_name" text,
    "research_sources_count" integer default 0,
    "last_research_date" timestamp with time zone,
    "additional_knowledge_enhanced" boolean,
    "additional_documents_count" bigint,
    "additional_enhancement_timestamp" timestamp without time zone,
    "rag_query_type" text,
    "selected" boolean default false,
    "affiliate_opportunities" json,
    "affiliate_program_ids" text,
    "affiliate_programs_data" json,
    "estimated_annual_revenue" bigint,
    "revenue_breakdown" json,
    "monetization_priority" text,
    "monetization_analysis" json,
    "'monetization_score" bigint,
    "monetization_score" text,
    "'rag_balance_emphasis': Inputs.rag_balance_emphasis," text
);


alter table "public"."Titles" enable row level security;

create table "public"."Titles_citations" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "title_id" text,
    "author" text,
    "formatted_text" text,
    "url" text,
    "publication_date" text,
    "citation_id" text,
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."Titles_citations" enable row level security;

create table "public"."Tones" (
    "Tone" text
);


alter table "public"."Tones" enable row level security;

create table "public"."affiliate_offers" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "workflow_session_id" text,
    "offer_name" text not null,
    "offer_description" text,
    "commission_rate" numeric(5,2),
    "access_instructions" text,
    "linkup_data" jsonb,
    "status" text default 'active'::text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "research_score" numeric(3,2) default 0.0,
    "relevance_score" numeric(3,2) default 0.0,
    "subtopics" text[] default '{}'::text[],
    "commission_range" character varying(20),
    "network_name" character varying(100),
    "program_url" text,
    "application_requirements" text,
    "payment_terms" text,
    "cookie_duration" integer,
    "last_verified" timestamp with time zone,
    "verification_status" character varying(20) default 'unverified'::character varying,
    "popularity_score" integer default 0,
    "conversion_rate" numeric(5,2),
    "avg_order_value" numeric(10,2),
    "target_audience" text[],
    "content_opportunities" jsonb default '[]'::jsonb,
    "seasonal_trends" jsonb default '{}'::jsonb,
    "competitor_analysis" jsonb default '{}'::jsonb
);


alter table "public"."affiliate_offers" enable row level security;

create table "public"."affiliate_programs" (
    "id" uuid not null default gen_random_uuid(),
    "program_name" character varying(255) not null,
    "company_name" character varying(255) not null,
    "description" text,
    "website_url" text,
    "network_name" character varying(100),
    "commission_rate" numeric(5,2),
    "commission_type" character varying(50),
    "cookie_duration" integer,
    "payment_terms" text,
    "application_requirements" text,
    "program_url" text,
    "contact_email" character varying(255),
    "status" character varying(20) default 'active'::character varying,
    "verification_status" character varying(20) default 'unverified'::character varying,
    "last_verified" timestamp with time zone,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "research_score" numeric(3,2) default 0.0,
    "popularity_score" integer default 0,
    "conversion_rate" numeric(5,2),
    "avg_order_value" numeric(10,2),
    "target_audience" text[],
    "content_opportunities" jsonb default '[]'::jsonb,
    "seasonal_trends" jsonb default '{}'::jsonb,
    "competitor_analysis" jsonb default '{}'::jsonb,
    "source" character varying(50) default 'manual'::character varying,
    "data_quality_score" numeric(3,2) default 0.0,
    "last_researched" timestamp with time zone,
    "research_count" integer default 0
);


alter table "public"."affiliate_programs" enable row level security;

create table "public"."affiliate_research" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "search_term" character varying(255) not null,
    "niche" character varying(100),
    "budget_range" character varying(50),
    "results" jsonb,
    "status" character varying(20) default 'pending'::character varying,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."affiliate_research" enable row level security;

create table "public"."application_settings" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "user_id" uuid,
    "youTubeKey" text,
    "unsplashKey" text,
    "prompt_intentAnalisys" text,
    "perplexityAI_key" text,
    "geminiKey" text,
    "chatGptKey" text,
    "stabilityAiKey" text,
    "fluxKey" text,
    "openAIKey" text,
    "imagen3Key" text,
    "ImagenAPIKey" jsonb,
    "geminiModel" text,
    "perplexityModel" text,
    "openAIModel" text,
    "geminiModelDisplay" text,
    "perplexityModelDisplay" text,
    "openAIModelDisplay" text,
    "GoogleSearchAPIKey" text,
    "GoogleSearchEngineId" text,
    "DeepSeekKey" text,
    "DeepSeekDisplay" text,
    "DeepSeekModel" text,
    "pexels_key" text,
    "claudeModel" text,
    "claudeKey" text,
    "claudeDisplay" text,
    "SD_Model" text,
    "imagenModel" text,
    "fluxModel" text,
    "linkupKey" text,
    "googleBooksApiKey" text default ''::text,
    "idea_generation_url" text,
    "rag_url" text,
    "kimiModel" text,
    "kimiKey" text,
    "kimiDisplay" text,
    "trends_url" text,
    "article_creation_url" text,
    "article_creation_key" text
);


alter table "public"."application_settings" enable row level security;

create table "public"."blog_generation_results" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "trend_analysis_id" uuid not null,
    "total_ideas_generated" integer default 0,
    "average_quality_score" numeric(5,2) default 0.0,
    "processing_time_seconds" numeric(8,2) default 0.0,
    "llm_provider" text default ''::text,
    "llm_model" text default ''::text,
    "strategic_insights" jsonb default '{}'::jsonb,
    "success_predictions" jsonb default '{}'::jsonb,
    "implementation_recommendations" jsonb default '[]'::jsonb,
    "ideas_by_source" jsonb default '{}'::jsonb,
    "quality_tier_distribution" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now()
);


alter table "public"."blog_generation_results" enable row level security;

create table "public"."blog_idea_keyword_assignments" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "blog_idea_id" uuid,
    "imported_keyword_id" uuid,
    "keyword_type" text not null,
    "assignment_score" numeric(5,2) default 0,
    "is_active" boolean default true,
    "assignment_metadata" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."blog_idea_keyword_assignments" enable row level security;

create table "public"."blog_idea_performance" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "blog_idea_id" uuid not null,
    "published_url" text,
    "published_date" date,
    "actual_word_count" integer,
    "page_views" integer default 0,
    "unique_visitors" integer default 0,
    "time_on_page_seconds" integer default 0,
    "bounce_rate" numeric(5,2) default 0.0,
    "social_shares" integer default 0,
    "backlinks_count" integer default 0,
    "email_signups" integer default 0,
    "lead_conversions" integer default 0,
    "revenue_attributed" numeric(10,2) default 0.0,
    "average_search_position" numeric(5,2),
    "organic_traffic_percentage" numeric(5,2) default 0.0,
    "featured_snippet_captured" boolean default false,
    "prediction_accuracy_score" integer,
    "actual_vs_predicted_performance" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."blog_idea_performance" enable row level security;

create table "public"."blog_idea_templates" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid,
    "template_name" text not null,
    "content_format" text not null,
    "target_audience" text default 'professional'::text,
    "description" text default ''::text,
    "outline_template" jsonb default '[]'::jsonb,
    "key_points_template" jsonb default '[]'::jsonb,
    "engagement_hooks_template" jsonb default '[]'::jsonb,
    "seo_guidelines" jsonb default '{}'::jsonb,
    "is_public" boolean default false,
    "is_system_template" boolean default false,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."blog_idea_templates" enable row level security;

create table "public"."blog_ideas" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "trend_analysis_id" uuid not null,
    "title" text not null,
    "description" text,
    "content_format" text not null,
    "difficulty_level" text not null,
    "estimated_word_count" integer default 2500,
    "estimated_reading_time" integer,
    "overall_quality_score" integer not null,
    "viral_potential_score" integer,
    "seo_optimization_score" integer,
    "audience_alignment_score" integer,
    "content_feasibility_score" integer,
    "business_impact_score" integer,
    "primary_keywords" jsonb default '[]'::jsonb,
    "secondary_keywords" jsonb default '[]'::jsonb,
    "featured_snippet_opportunity" boolean default false,
    "outline" jsonb default '[]'::jsonb,
    "key_points" jsonb default '[]'::jsonb,
    "engagement_hooks" jsonb default '[]'::jsonb,
    "visual_elements" jsonb default '[]'::jsonb,
    "call_to_action" text,
    "business_value" text,
    "performance_estimates" jsonb default '{}'::jsonb,
    "generation_source" text,
    "source_topic_id" uuid,
    "source_opportunity_id" uuid,
    "selected" boolean default false,
    "priority_level" text default 'medium'::text,
    "scheduled_publish_date" date,
    "notes" text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "enhanced_primary_keywords" jsonb default '[]'::jsonb,
    "enhanced_secondary_keywords" jsonb default '[]'::jsonb,
    "keyword_research_data" jsonb default '{}'::jsonb,
    "traffic_potential_score" integer default 0,
    "competition_score" integer default 0,
    "keyword_suggestions" jsonb default '{}'::jsonb,
    "content_optimization_tips" jsonb default '[]'::jsonb,
    "keyword_research_enhanced" boolean default false,
    "keyword_source_tools" jsonb default '[]'::jsonb,
    "enhancement_timestamp" timestamp with time zone
);


alter table "public"."blog_ideas" enable row level security;

create table "public"."categoriesByPost" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "CategoryID" text,
    "Description" text,
    "user_id" uuid default gen_random_uuid(),
    "titleId" text,
    "domain" text,
    "sectionNumberingFlag" text,
    "tableOfContentsFlag" text
);


alter table "public"."categoriesByPost" enable row level security;

create table "public"."competitive_intelligence" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "competitive_data" jsonb not null,
    "created_at" timestamp with time zone default now(),
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."competitive_intelligence" enable row level security;

create table "public"."content_calendar" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "trend_analysis_id" uuid not null,
    "publishing_strategy" jsonb not null,
    "priority_scheduling" jsonb not null,
    "seasonal_optimization" jsonb default '{}'::jsonb,
    "content_series_opportunities" jsonb default '[]'::jsonb,
    "format_distribution" jsonb default '{}'::jsonb,
    "estimated_resource_requirements" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."content_calendar" enable row level security;

create table "public"."content_ideas" (
    "id" uuid not null default gen_random_uuid(),
    "title" text not null,
    "description" text,
    "content_type" character varying(20) not null,
    "category" character varying(50) not null,
    "subtopic" text not null,
    "topic_id" uuid not null,
    "user_id" uuid not null,
    "keywords" text[],
    "seo_score" integer default 0,
    "difficulty_level" character varying(20) default 'medium'::character varying,
    "estimated_read_time" integer,
    "target_audience" text,
    "content_angle" text,
    "monetization_potential" character varying(20) default 'medium'::character varying,
    "technical_complexity" character varying(20) default 'medium'::character varying,
    "development_effort" character varying(20) default 'medium'::character varying,
    "market_demand" character varying(20) default 'medium'::character varying,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "status" character varying(50) default 'draft'::character varying,
    "published" boolean default false,
    "published_at" timestamp with time zone,
    "published_to_titles" boolean default false,
    "titles_record_id" uuid,
    "priority" character varying(50) default 'medium'::character varying,
    "workflow_status" character varying(50) default 'idea_generated'::character varying,
    "content_generated" boolean default false,
    "content_brief_generated" boolean default false,
    "overall_quality_score" integer default 0,
    "seo_optimization_score" integer default 0,
    "traffic_potential_score" integer default 0,
    "viral_potential_score" integer default 0,
    "competition_score" integer default 0,
    "content_outline" jsonb default '[]'::jsonb,
    "key_points" jsonb default '[]'::jsonb,
    "primary_keywords" jsonb default '[]'::jsonb,
    "secondary_keywords" jsonb default '[]'::jsonb,
    "enhanced_keywords" jsonb default '[]'::jsonb,
    "keyword_research_data" jsonb default '{}'::jsonb,
    "keyword_research_enhanced" boolean default false,
    "affiliate_opportunities" jsonb default '{}'::jsonb,
    "monetization_score" integer default 0,
    "estimated_annual_revenue" numeric(10,2) default 0,
    "monetization_priority" character varying(50) default 'medium'::character varying,
    "generation_method" character varying(50) default 'llm'::character varying,
    "generation_prompt" text,
    "generation_parameters" jsonb default '{}'::jsonb,
    "enhancement_timestamp" timestamp with time zone,
    "estimated_word_count" integer default 2500,
    "estimated_reading_time" integer default 12
);


alter table "public"."content_ideas" enable row level security;

create table "public"."content_opportunities" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "title" character varying(500) not null,
    "format" character varying(100),
    "difficulty" integer,
    "engagement_potential" character varying(50),
    "selected" boolean default false,
    "additional_data" text,
    "created_at" timestamp with time zone,
    "user_id" uuid
);


alter table "public"."content_opportunities" enable row level security;

create table "public"."embeddings" (
    "id" bigint generated always as identity not null,
    "content" text not null,
    "embedding" extensions.vector(384),
    "sequence" text,
    "cluster" boolean default false,
    "title" text,
    "user_id" uuid,
    "source_id" bigint,
    "created_at" timestamp without time zone default now(),
    "metadata" jsonb,
    "vector" extensions.vector(384),
    "articleSummary" boolean default false
);


alter table "public"."embeddings" enable row level security;

create table "public"."geographic_insights" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "geographic_data" jsonb not null,
    "created_at" timestamp with time zone default now(),
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."geographic_insights" enable row level security;

create table "public"."imported_keywords" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "session_id" uuid,
    "keyword" text not null,
    "search_volume" integer default 0,
    "keyword_difficulty" numeric(5,2) default 0,
    "cpc" numeric(8,2) default 0,
    "competition" text default 'medium'::text,
    "search_intent" text default 'informational'::text,
    "trend" text default 'stable'::text,
    "related_keywords" jsonb default '[]'::jsonb,
    "source_tool" text not null,
    "opportunity_score" integer default 0,
    "is_selected" boolean default false,
    "keyword_metadata" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."imported_keywords" enable row level security;

create table "public"."indexed_documents" (
    "id" integer not null default nextval('indexed_documents_id_seq'::regclass),
    "collection_name" text not null,
    "title" text not null,
    "content" text not null
);


alter table "public"."indexed_documents" enable row level security;

create table "public"."infographic" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "name" text,
    "CSS" text,
    "HTML" text,
    "numberOfItems" smallint,
    "JS" text,
    "prompt" text,
    "jsonStructure" text,
    "type" text,
    "height" bigint default '600'::bigint,
    "width" bigint default '800'::bigint,
    "clipX" bigint,
    "clipY" bigint,
    "clipWidth" bigint,
    "clipHeight" bigint,
    "sampleImage" text
);


alter table "public"."infographic" enable row level security;

create table "public"."infographicDetails" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "name" text,
    "maxCharacters" smallint,
    "infographicId" bigint
);


alter table "public"."infographicDetails" enable row level security;

create table "public"."keyword_intelligence" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "high_volume_keywords" text[],
    "low_competition_keywords" text[],
    "emerging_keywords" text[],
    "additional_data" text,
    "created_at" timestamp without time zone,
    "user_id" uuid
);


alter table "public"."keyword_intelligence" enable row level security;

create table "public"."keyword_opportunities_reports" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "session_id" uuid,
    "analysis_id" uuid,
    "report_name" text,
    "total_keywords_analyzed" integer default 0,
    "high_opportunity_count" integer default 0,
    "medium_opportunity_count" integer default 0,
    "low_opportunity_count" integer default 0,
    "total_search_volume" bigint default 0,
    "average_difficulty" numeric(5,2) default 0,
    "average_cpc" numeric(8,2) default 0,
    "report_data" jsonb default '{}'::jsonb,
    "insights" jsonb default '[]'::jsonb,
    "recommendations" jsonb default '[]'::jsonb,
    "content_recommendations" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."keyword_opportunities_reports" enable row level security;

create table "public"."keyword_research_sessions" (
    "id" uuid not null default extensions.uuid_generate_v4(),
    "user_id" uuid not null,
    "session_name" text,
    "source_tool" text not null,
    "filename" text,
    "total_keywords_imported" integer default 0,
    "keywords_validated" integer default 0,
    "validation_status" text default 'pending'::text,
    "validation_issues" jsonb default '[]'::jsonb,
    "validation_warnings" jsonb default '[]'::jsonb,
    "session_metadata" jsonb default '{}'::jsonb,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."keyword_research_sessions" enable row level security;

create table "public"."keywords" (
    "id" uuid not null default gen_random_uuid(),
    "keyword" text not null,
    "search_volume" integer,
    "difficulty" numeric(5,2),
    "cpc" numeric(10,4),
    "topic_id" uuid not null,
    "user_id" uuid not null,
    "source" character varying(20) not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."keywords" enable row level security;

create table "public"."lindex_collections" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "name" text,
    "user_id" uuid,
    "display_name" text default ''::text,
    "dimensions" text default ''::text,
    "Description" text
);


alter table "public"."lindex_collections" enable row level security;

create table "public"."lindex_documents" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "title" text,
    "user_id" uuid,
    "parsedText" text,
    "author" text,
    "summary_text" text,
    "refined_text" text,
    "collectionId" bigint,
    "in_vector_store" boolean default false,
    "summary_medium" text,
    "chunk_stats" text,
    "last_processed" text,
    "total_chunks" text,
    "processed_chunks" text,
    "failed_chunks" text,
    "avg_chunk_size" text,
    "processing_status" text,
    "summary_short" text,
    "source_type" text,
    "url" text,
    "selected" boolean,
    "scraped" boolean,
    "web_content" text,
    "doc_size" integer,
    "chunk_count" integer,
    "importance_score" double precision,
    "processing_metadata" jsonb,
    "summary_error" text,
    "error_message" text,
    "error_details" text,
    "file_type" text default 'unknown'::text,
    "filename" text,
    "upload_date" text,
    "title_id" text,
    "focus_keyword" text,
    "research_timestamp" timestamp without time zone,
    "data_source_type" text,
    "enhancement_type" text default 'original'::text,
    "publish_date" text
);


alter table "public"."lindex_documents" enable row level security;

create table "public"."lindex_embedding_chunk" (
    "id" integer generated always as identity not null,
    "vec" extensions.vector(1536),
    "metadata" jsonb,
    "content" text,
    "docid" text,
    "section_id" text
);


alter table "public"."lindex_embedding_chunk" enable row level security;

create table "public"."lindex_sections" (
    "id" integer not null default nextval('documents_id_seq'::regclass),
    "section_content" text not null,
    "section_summary" text,
    "section_title" text,
    "section_order" numeric,
    "docid" text,
    "section" text,
    "user_id" uuid,
    "chunk_order" numeric,
    "doc_identifier" text,
    "metadata" text
);


alter table "public"."lindex_sections" enable row level security;

create table "public"."llm_configurations" (
    "id" uuid not null default gen_random_uuid(),
    "provider_id" uuid not null,
    "name" character varying(255) not null,
    "config_data" jsonb not null,
    "is_active" boolean default true,
    "is_default" boolean default false,
    "priority" integer default 0,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."llm_configurations" enable row level security;

create table "public"."llm_providers" (
    "id" uuid not null default gen_random_uuid(),
    "name" character varying(255) not null,
    "provider_type" character varying(50) not null,
    "model_name" character varying(255) not null,
    "api_key_env_var" character varying(100),
    "base_url" text,
    "api_version" character varying(50),
    "max_tokens" integer default 2000,
    "temperature" numeric(3,2) default 0.7,
    "top_p" numeric(3,2) default 1.0,
    "frequency_penalty" numeric(3,2) default 0.0,
    "presence_penalty" numeric(3,2) default 0.0,
    "cost_per_1k_tokens" numeric(10,6) default 0.0,
    "max_requests_per_minute" integer default 60,
    "average_response_time_ms" integer default 2000,
    "is_active" boolean default true,
    "is_default" boolean default false,
    "priority" integer default 0,
    "custom_config" jsonb,
    "last_used" timestamp with time zone,
    "total_requests" integer default 0,
    "successful_requests" integer default 0,
    "failed_requests" integer default 0,
    "total_tokens_used" integer default 0,
    "total_cost" numeric(10,4) default 0.0,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."llm_providers" enable row level security;

create table "public"."manual_action_suggestions" (
    "id" uuid not null default gen_random_uuid(),
    "title_id" text not null,
    "action_type" text not null,
    "title" text not null,
    "description" text not null,
    "resource_name" text not null,
    "estimated_effort_hours" integer not null,
    "difficulty_level" text not null,
    "expected_benefit" text not null,
    "cost_estimate" text not null,
    "implementation_notes" text not null,
    "content_enhancement_potential" text not null,
    "impact_score" integer not null,
    "feasibility_score" integer not null,
    "priority_score" numeric(5,2) not null,
    "priority_level" text not null,
    "status" text not null default 'suggested'::text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "completed_at" timestamp with time zone,
    "research_context" jsonb,
    "additional_data" jsonb,
    "notes" text,
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."manual_action_suggestions" enable row level security;

create table "public"."mySources" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "title" text,
    "user_id" uuid,
    "cluster" boolean default false,
    "content" text default ''::text,
    "url" text default ''::text,
    "summary" text,
    "summarized" boolean default false
);


alter table "public"."mySources" enable row level security;

create table "public"."offer_analytics" (
    "id" uuid not null default gen_random_uuid(),
    "offer_id" uuid not null,
    "user_id" uuid not null,
    "research_session_id" uuid,
    "click_count" integer default 0,
    "conversion_count" integer default 0,
    "revenue_generated" numeric(10,2) default 0.0,
    "commission_earned" numeric(10,2) default 0.0,
    "time_spent_seconds" integer default 0,
    "selection_count" integer default 0,
    "content_ideas_generated" integer default 0,
    "first_viewed" timestamp with time zone default now(),
    "last_viewed" timestamp with time zone default now(),
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."offer_analytics" enable row level security;

create table "public"."offer_research_sessions" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "session_name" character varying(255),
    "search_terms" text[] not null,
    "research_scope" character varying(50) default 'comprehensive'::character varying,
    "llm_analysis" jsonb,
    "discovered_programs" uuid[] default '{}'::uuid[],
    "selected_offers" uuid[] default '{}'::uuid[],
    "research_quality_score" numeric(3,2),
    "status" character varying(20) default 'active'::character varying,
    "created_at" timestamp with time zone default now(),
    "completed_at" timestamp with time zone
);


alter table "public"."offer_research_sessions" enable row level security;

create table "public"."postTypes" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "postType" text,
    "Outline" text,
    "outlinePrompt" text,
    "contentPrompt" text,
    "textSearchMatchType" text,
    "searchPrompt" text,
    "Outline_Prompt Part 1" text,
    "Outline_Prompt Part 2" text,
    "Content_PromptPart1" text,
    "Content_PromptPart2" text
);


alter table "public"."postTypes" enable row level security;

create table "public"."research_program_links" (
    "id" uuid not null default gen_random_uuid(),
    "research_session_id" uuid,
    "program_id" uuid,
    "link_type" text,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."research_program_links" enable row level security;

create table "public"."research_topics" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "title" character varying(255) not null,
    "description" text,
    "status" character varying(50) not null default 'active'::character varying,
    "created_at" timestamp with time zone not null default now(),
    "updated_at" timestamp with time zone not null default now(),
    "version" integer not null default 1
);


alter table "public"."research_topics" enable row level security;

create table "public"."seasonal_calendar" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "seasonal_data" jsonb not null,
    "created_at" timestamp with time zone default now(),
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."seasonal_calendar" enable row level security;

create table "public"."sectionSpecificPrompts" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "postTypeID" bigint,
    "sectionHeader" text,
    "prompt" text,
    "PostType" text
);


alter table "public"."sectionSpecificPrompts" enable row level security;

create table "public"."summaries" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "docid" text,
    "section_title" text,
    "summary" text,
    "sequence" text,
    "section_order" text
);


alter table "public"."summaries" enable row level security;

create table "public"."topic_decompositions" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "search_query" text not null,
    "subtopics" text[] not null,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now(),
    "research_topic_id" uuid,
    "original_topic_included" boolean not null default true
);


alter table "public"."topic_decompositions" enable row level security;

create table "public"."trend_analyses" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid,
    "topic" character varying(255) not null,
    "target_audience" character varying(255),
    "focus_area" character varying(255),
    "created_at" timestamp without time zone default now(),
    "status" character varying(50) default 'completed'::character varying,
    "metadata" jsonb,
    "updated_at" timestamp without time zone,
    "affiliate_research_id" uuid
);


alter table "public"."trend_analyses" enable row level security;

create table "public"."trend_analysis" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "keywords" text[] not null,
    "analysis_data" jsonb,
    "time_period" character varying(50),
    "status" character varying(20) default 'pending'::character varying,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."trend_analysis" enable row level security;

create table "public"."trend_predictions" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "predictions_data" jsonb not null,
    "created_at" timestamp with time zone default now(),
    "user_id" uuid default gen_random_uuid()
);


alter table "public"."trend_predictions" enable row level security;

create table "public"."trending_topics" (
    "id" uuid not null default gen_random_uuid(),
    "trend_analysis_id" uuid,
    "title" character varying(500) not null,
    "viral_potential" integer,
    "keywords" text[],
    "search_volume" character varying(50),
    "competition" character varying(50),
    "selected" boolean default false,
    "additional_data" text,
    "created_at" timestamp without time zone,
    "user_id" uuid
);


alter table "public"."trending_topics" enable row level security;

create table "public"."user_offer_preferences" (
    "id" uuid not null default gen_random_uuid(),
    "user_id" uuid not null,
    "preferred_networks" text[] default '{}'::text[],
    "preferred_commission_ranges" text[] default '{}'::text[],
    "preferred_categories" text[] default '{}'::text[],
    "preferred_difficulty_levels" text[] default '{}'::text[],
    "successful_offers" uuid[] default '{}'::uuid[],
    "rejected_offers" uuid[] default '{}'::uuid[],
    "content_preferences" jsonb default '{}'::jsonb,
    "learning_enabled" boolean default true,
    "last_updated" timestamp with time zone default now(),
    "created_at" timestamp with time zone default now()
);


alter table "public"."user_offer_preferences" enable row level security;

create table "public"."user_profile" (
    "id" uuid not null,
    "created_at" timestamp with time zone not null default now(),
    "email" character varying,
    "first_name" text,
    "last_name" text,
    "DalleKey" text,
    "GeminiKey" text,
    "StabilityAIKey" text,
    "selectedImageGen" text,
    "selectedTextGen" text,
    "WordPress_ID" text,
    "unsplashKey" text,
    "chatGptKey" text,
    "writingStyle" text,
    "authorSummary" text
);


alter table "public"."user_profile" enable row level security;

create table "public"."users" (
    "id" uuid not null default gen_random_uuid(),
    "email" character varying(255) not null,
    "password_hash" character varying(255) not null,
    "first_name" character varying(100),
    "last_name" character varying(100),
    "role" character varying(20) default 'user'::character varying,
    "is_active" boolean default true,
    "is_verified" boolean default false,
    "email_verification_token" character varying(255),
    "email_verification_expires" timestamp with time zone,
    "last_login" timestamp with time zone,
    "failed_login_attempts" integer default 0,
    "locked_until" timestamp with time zone,
    "created_at" timestamp with time zone default now(),
    "updated_at" timestamp with time zone default now()
);


alter table "public"."users" enable row level security;

create table "public"."wordPress_details" (
    "id" bigint generated by default as identity not null,
    "created_at" timestamp with time zone not null default now(),
    "user_id" uuid default gen_random_uuid(),
    "domain" text,
    "wordpress_key" text,
    "app_name" text,
    "wpUserName" text,
    "websiteDescription" text,
    "targetAudienceDescription" text
);


alter table "public"."wordPress_details" enable row level security;

alter table "public"."api_keys" add column "environment" character varying(20) default 'production'::character varying;

alter table "public"."api_keys" add column "key_name" character varying(100) not null;

alter table "public"."api_keys" alter column "base_url" drop not null;

alter table "public"."api_keys" alter column "base_url" set data type text using "base_url"::text;

alter table "public"."api_keys" alter column "created_at" drop not null;

alter table "public"."api_keys" alter column "id" set default gen_random_uuid();

alter table "public"."api_keys" alter column "is_active" drop not null;

alter table "public"."api_keys" alter column "updated_at" drop not null;

alter table "public"."api_keys" enable row level security;

alter table "public"."dataforseo_api_logs" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."keyword_research_data" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."subtopic_suggestions" alter column "id" set default extensions.uuid_generate_v4();

alter table "public"."trend_analysis_data" alter column "id" set default extensions.uuid_generate_v4();

alter sequence "public"."documents_id_seq" owned by "public"."lindex_sections"."id";

alter sequence "public"."indexed_documents_id_seq" owned by "public"."indexed_documents"."id";

CREATE UNIQUE INDEX "Books_pkey" ON public.lindex_documents USING btree (id);

CREATE UNIQUE INDEX "CategoriesByPost_pkey" ON public."categoriesByPost" USING btree (id);

CREATE UNIQUE INDEX "Infographics_pkey" ON public.infographic USING btree (id);

CREATE UNIQUE INDEX "KnowledgeBase_pkey" ON public."mySources" USING btree (id);

CREATE UNIQUE INDEX "PlannedArticles_pkey" ON public."PlannedArticles" USING btree (id);

CREATE UNIQUE INDEX "RSS_pkey" ON public."RSS" USING btree (id);

CREATE UNIQUE INDEX "Summaries_pkey" ON public.summaries USING btree (id);

CREATE UNIQUE INDEX "TableOfContents_pkey" ON public."TableOfContents" USING btree (id);

CREATE UNIQUE INDEX "Titles_citations_pkey" ON public."Titles_citations" USING btree (id);

CREATE UNIQUE INDEX "Titles_pkey" ON public."Titles" USING btree (id);

CREATE UNIQUE INDEX "WordPress_details_pkey" ON public."wordPress_details" USING btree (id);

CREATE UNIQUE INDEX affiliate_offers_pkey ON public.affiliate_offers USING btree (id);

CREATE UNIQUE INDEX affiliate_programs_pkey ON public.affiliate_programs USING btree (id);

CREATE UNIQUE INDEX affiliate_research_pkey ON public.affiliate_research USING btree (id);

CREATE UNIQUE INDEX api_keys_key_name_key ON public.api_keys USING btree (key_name);

CREATE UNIQUE INDEX application_settings_pkey ON public.application_settings USING btree (id);

CREATE UNIQUE INDEX blog_generation_results_pkey ON public.blog_generation_results USING btree (id);

CREATE UNIQUE INDEX blog_idea_keyword_assignments_blog_idea_id_imported_keyword_key ON public.blog_idea_keyword_assignments USING btree (blog_idea_id, imported_keyword_id, keyword_type);

CREATE UNIQUE INDEX blog_idea_keyword_assignments_pkey ON public.blog_idea_keyword_assignments USING btree (id);

CREATE UNIQUE INDEX blog_idea_performance_pkey ON public.blog_idea_performance USING btree (id);

CREATE UNIQUE INDEX blog_idea_templates_pkey ON public.blog_idea_templates USING btree (id);

CREATE UNIQUE INDEX blog_ideas_pkey ON public.blog_ideas USING btree (id);

CREATE UNIQUE INDEX competitive_intelligence_pkey ON public.competitive_intelligence USING btree (id);

CREATE UNIQUE INDEX content_calendar_pkey ON public.content_calendar USING btree (id);

CREATE UNIQUE INDEX content_ideas_pkey ON public.content_ideas USING btree (id);

CREATE UNIQUE INDEX content_opportunities_pkey ON public.content_opportunities USING btree (id);

CREATE UNIQUE INDEX documents_pkey ON public.lindex_sections USING btree (id);

CREATE INDEX embeddings_embedding_idx ON public.embeddings USING hnsw (embedding extensions.vector_ip_ops);

CREATE INDEX embeddings_embedding_idx1 ON public.embeddings USING ivfflat (embedding) WITH (lists='100');

CREATE UNIQUE INDEX embeddings_pkey ON public.embeddings USING btree (id);

CREATE UNIQUE INDEX geographic_insights_pkey ON public.geographic_insights USING btree (id);

CREATE INDEX idx_affiliate_offers_last_verified ON public.affiliate_offers USING btree (last_verified);

CREATE INDEX idx_affiliate_offers_popularity_score ON public.affiliate_offers USING btree (popularity_score DESC);

CREATE INDEX idx_affiliate_offers_research_score ON public.affiliate_offers USING btree (research_score DESC);

CREATE INDEX idx_affiliate_offers_verification_status ON public.affiliate_offers USING btree (verification_status);

CREATE INDEX idx_affiliate_programs_commission_rate ON public.affiliate_programs USING btree (commission_rate DESC);

CREATE INDEX idx_affiliate_programs_network_name ON public.affiliate_programs USING btree (network_name);

CREATE INDEX idx_affiliate_programs_research_score ON public.affiliate_programs USING btree (research_score DESC);

CREATE INDEX idx_affiliate_programs_status ON public.affiliate_programs USING btree (status);

CREATE INDEX idx_affiliate_programs_verification_status ON public.affiliate_programs USING btree (verification_status);

CREATE INDEX idx_affiliate_research_user_id ON public.affiliate_research USING btree (user_id);

CREATE INDEX idx_api_keys_active ON public.api_keys USING btree (is_active);

CREATE INDEX idx_api_keys_provider ON public.api_keys USING btree (provider);

CREATE INDEX idx_blog_generation_results_created_at ON public.blog_generation_results USING btree (created_at DESC);

CREATE INDEX idx_blog_generation_results_trend_analysis_id ON public.blog_generation_results USING btree (trend_analysis_id);

CREATE INDEX idx_blog_generation_results_user_id ON public.blog_generation_results USING btree (user_id);

CREATE INDEX idx_blog_ideas_competition_score ON public.blog_ideas USING btree (competition_score);

CREATE INDEX idx_blog_ideas_content_format ON public.blog_ideas USING btree (content_format);

CREATE INDEX idx_blog_ideas_keyword_enhanced ON public.blog_ideas USING btree (keyword_research_enhanced);

CREATE INDEX idx_blog_ideas_priority ON public.blog_ideas USING btree (priority_level);

CREATE INDEX idx_blog_ideas_quality_score ON public.blog_ideas USING btree (overall_quality_score DESC);

CREATE INDEX idx_blog_ideas_scheduled_date ON public.blog_ideas USING btree (scheduled_publish_date) WHERE (scheduled_publish_date IS NOT NULL);

CREATE INDEX idx_blog_ideas_selected ON public.blog_ideas USING btree (selected) WHERE (selected = true);

CREATE INDEX idx_blog_ideas_traffic_potential ON public.blog_ideas USING btree (traffic_potential_score);

CREATE INDEX idx_blog_ideas_traffic_score ON public.blog_ideas USING btree (traffic_potential_score);

CREATE INDEX idx_blog_ideas_trend_analysis_id ON public.blog_ideas USING btree (trend_analysis_id);

CREATE INDEX idx_blog_ideas_user_id ON public.blog_ideas USING btree (user_id);

CREATE INDEX idx_blog_keyword_assignments_blog_idea_id ON public.blog_idea_keyword_assignments USING btree (blog_idea_id);

CREATE INDEX idx_blog_keyword_assignments_keyword_id ON public.blog_idea_keyword_assignments USING btree (imported_keyword_id);

CREATE INDEX idx_blog_keyword_assignments_type ON public.blog_idea_keyword_assignments USING btree (keyword_type);

CREATE INDEX idx_blog_keyword_assignments_user_id ON public.blog_idea_keyword_assignments USING btree (user_id);

CREATE INDEX idx_blog_performance_blog_idea_id ON public.blog_idea_performance USING btree (blog_idea_id);

CREATE INDEX idx_blog_performance_page_views ON public.blog_idea_performance USING btree (page_views DESC);

CREATE INDEX idx_blog_performance_published_date ON public.blog_idea_performance USING btree (published_date DESC);

CREATE INDEX idx_blog_performance_user_id ON public.blog_idea_performance USING btree (user_id);

CREATE INDEX idx_blog_templates_format ON public.blog_idea_templates USING btree (content_format);

CREATE INDEX idx_blog_templates_public ON public.blog_idea_templates USING btree (is_public) WHERE (is_public = true);

CREATE INDEX idx_blog_templates_user_id ON public.blog_idea_templates USING btree (user_id);

CREATE INDEX idx_competitive_intelligence_analysis_id ON public.competitive_intelligence USING btree (trend_analysis_id);

CREATE INDEX idx_content_calendar_trend_analysis_id ON public.content_calendar USING btree (trend_analysis_id);

CREATE INDEX idx_content_calendar_user_id ON public.content_calendar USING btree (user_id);

CREATE INDEX idx_content_ideas_category ON public.content_ideas USING btree (category);

CREATE INDEX idx_content_ideas_content_type ON public.content_ideas USING btree (content_type);

CREATE INDEX idx_content_ideas_created_at ON public.content_ideas USING btree (created_at);

CREATE INDEX idx_content_ideas_priority ON public.content_ideas USING btree (priority);

CREATE INDEX idx_content_ideas_published ON public.content_ideas USING btree (published);

CREATE INDEX idx_content_ideas_published_at ON public.content_ideas USING btree (published_at);

CREATE INDEX idx_content_ideas_published_to_titles ON public.content_ideas USING btree (published_to_titles);

CREATE INDEX idx_content_ideas_status ON public.content_ideas USING btree (status);

CREATE INDEX idx_content_ideas_subtopic ON public.content_ideas USING btree (subtopic);

CREATE INDEX idx_content_ideas_topic_id ON public.content_ideas USING btree (topic_id);

CREATE INDEX idx_content_ideas_type_category ON public.content_ideas USING btree (content_type, category);

CREATE INDEX idx_content_ideas_user_id ON public.content_ideas USING btree (user_id);

CREATE INDEX idx_content_ideas_user_topic ON public.content_ideas USING btree (user_id, topic_id);

CREATE INDEX idx_content_ideas_workflow_status ON public.content_ideas USING btree (workflow_status);

CREATE INDEX idx_dataforseo_api_logs_status_code ON public.dataforseo_api_logs USING btree (status_code);

CREATE INDEX idx_geographic_insights_analysis_id ON public.geographic_insights USING btree (trend_analysis_id);

CREATE INDEX idx_imported_keywords_difficulty ON public.imported_keywords USING btree (keyword_difficulty);

CREATE INDEX idx_imported_keywords_keyword ON public.imported_keywords USING btree (keyword);

CREATE INDEX idx_imported_keywords_opportunity_score ON public.imported_keywords USING btree (opportunity_score);

CREATE INDEX idx_imported_keywords_search_volume ON public.imported_keywords USING btree (search_volume);

CREATE INDEX idx_imported_keywords_session_id ON public.imported_keywords USING btree (session_id);

CREATE INDEX idx_imported_keywords_user_id ON public.imported_keywords USING btree (user_id);

CREATE INDEX idx_keyword_reports_analysis_id ON public.keyword_opportunities_reports USING btree (analysis_id);

CREATE INDEX idx_keyword_reports_session_id ON public.keyword_opportunities_reports USING btree (session_id);

CREATE INDEX idx_keyword_reports_user_id ON public.keyword_opportunities_reports USING btree (user_id);

CREATE INDEX idx_keyword_research_data_composite ON public.keyword_research_data USING btree (intent_type, keyword_difficulty, search_volume, updated_at);

CREATE INDEX idx_keyword_research_data_intent_type ON public.keyword_research_data USING btree (intent_type);

CREATE INDEX idx_keyword_research_data_priority_score ON public.keyword_research_data USING btree (priority_score);

CREATE INDEX idx_keywords_created_at ON public.keywords USING btree (created_at);

CREATE INDEX idx_keywords_source ON public.keywords USING btree (source);

CREATE INDEX idx_keywords_topic_id ON public.keywords USING btree (topic_id);

CREATE INDEX idx_keywords_user_id ON public.keywords USING btree (user_id);

CREATE INDEX idx_keywords_user_topic ON public.keywords USING btree (user_id, topic_id);

CREATE INDEX idx_lindex_documents_data_source_type ON public.lindex_documents USING btree (data_source_type);

CREATE INDEX idx_lindex_documents_enhancement_type ON public.lindex_documents USING btree (enhancement_type);

CREATE INDEX idx_lindex_documents_processing_status ON public.lindex_documents USING btree (processing_status);

CREATE INDEX idx_lindex_documents_source_type_enhancement ON public.lindex_documents USING btree (source_type, enhancement_type);

CREATE INDEX idx_lindex_documents_source_type_title_id ON public.lindex_documents USING btree (source_type, title_id);

CREATE INDEX idx_lindex_documents_title_id ON public.lindex_documents USING btree (title_id);

CREATE INDEX idx_llm_configurations_is_active ON public.llm_configurations USING btree (is_active);

CREATE INDEX idx_llm_configurations_provider_id ON public.llm_configurations USING btree (provider_id);

CREATE INDEX idx_llm_providers_is_active ON public.llm_providers USING btree (is_active);

CREATE INDEX idx_llm_providers_is_default ON public.llm_providers USING btree (is_default);

CREATE INDEX idx_llm_providers_priority ON public.llm_providers USING btree (priority);

CREATE INDEX idx_llm_providers_provider_type ON public.llm_providers USING btree (provider_type);

CREATE INDEX idx_manual_action_suggestions_action_type ON public.manual_action_suggestions USING btree (action_type);

CREATE INDEX idx_manual_action_suggestions_priority_level ON public.manual_action_suggestions USING btree (priority_level);

CREATE INDEX idx_manual_action_suggestions_priority_score ON public.manual_action_suggestions USING btree (priority_score DESC);

CREATE INDEX idx_manual_action_suggestions_status ON public.manual_action_suggestions USING btree (status);

CREATE INDEX idx_manual_action_suggestions_title_id ON public.manual_action_suggestions USING btree (title_id);

CREATE INDEX idx_offer_analytics_offer_id ON public.offer_analytics USING btree (offer_id);

CREATE INDEX idx_offer_analytics_revenue_generated ON public.offer_analytics USING btree (revenue_generated DESC);

CREATE INDEX idx_offer_analytics_user_id ON public.offer_analytics USING btree (user_id);

CREATE INDEX idx_offer_research_sessions_created_at ON public.offer_research_sessions USING btree (created_at DESC);

CREATE INDEX idx_offer_research_sessions_status ON public.offer_research_sessions USING btree (status);

CREATE INDEX idx_offer_research_sessions_user_id ON public.offer_research_sessions USING btree (user_id);

CREATE INDEX idx_research_program_links_program ON public.research_program_links USING btree (program_id);

CREATE INDEX idx_research_program_links_session ON public.research_program_links USING btree (research_session_id);

CREATE INDEX idx_research_topics_created_at ON public.research_topics USING btree (created_at);

CREATE INDEX idx_research_topics_status ON public.research_topics USING btree (status);

CREATE INDEX idx_research_topics_user_id ON public.research_topics USING btree (user_id);

CREATE UNIQUE INDEX idx_research_topics_user_title ON public.research_topics USING btree (user_id, title);

CREATE INDEX idx_seasonal_calendar_analysis_id ON public.seasonal_calendar USING btree (trend_analysis_id);

CREATE INDEX idx_subtopic_suggestions_growth_potential ON public.subtopic_suggestions USING btree (growth_potential);

CREATE INDEX idx_titles_additional_enhanced ON public."Titles" USING btree (additional_knowledge_enhanced);

CREATE INDEX idx_titles_blog_idea_id ON public."Titles" USING btree (blog_idea_id);

CREATE INDEX idx_titles_focus_keyword ON public."Titles" USING btree (focus_keyword);

CREATE INDEX idx_titles_gap_status ON public."Titles" USING btree (status, knowledge_gaps_closed);

CREATE INDEX idx_titles_gaps_closed ON public."Titles" USING btree (knowledge_gaps_closed);

CREATE INDEX idx_titles_priority_level ON public."Titles" USING btree (priority_level);

CREATE INDEX idx_titles_published ON public."Titles" USING btree (published);

CREATE INDEX idx_titles_scheduled_publish_date ON public."Titles" USING btree (scheduled_publish_date);

CREATE INDEX idx_titles_seo_optimization_score ON public."Titles" USING btree (seo_optimization_score);

CREATE INDEX idx_titles_status_gaps ON public."Titles" USING btree (status, knowledge_gaps_closed);

CREATE INDEX idx_titles_trend_analysis_id ON public."Titles" USING btree (trend_analysis_id);

CREATE INDEX idx_titles_user_id ON public."Titles" USING btree (user_id);

CREATE INDEX idx_titles_workflow ON public."Titles" USING btree (workflow_status, knowledge_enhanced);

CREATE INDEX idx_titles_workflow_status ON public."Titles" USING btree (workflow_status);

CREATE INDEX idx_titles_wp_status ON public."Titles" USING btree (wp_status);

CREATE INDEX idx_topic_decompositions_research_topic_id ON public.topic_decompositions USING btree (research_topic_id);

CREATE UNIQUE INDEX idx_topic_decompositions_research_topic_search ON public.topic_decompositions USING btree (research_topic_id, search_query);

CREATE INDEX idx_trend_analyses_affiliate_research ON public.trend_analyses USING btree (affiliate_research_id);

CREATE INDEX idx_trend_analysis_data_composite ON public.trend_analysis_data USING btree (subtopic, location, time_range, updated_at);

CREATE INDEX idx_trend_analysis_data_time_range ON public.trend_analysis_data USING btree (time_range);

CREATE INDEX idx_trend_analysis_user_id ON public.trend_analysis USING btree (user_id);

CREATE INDEX idx_trend_predictions_analysis_id ON public.trend_predictions USING btree (trend_analysis_id);

CREATE INDEX idx_user_offer_preferences_user_id ON public.user_offer_preferences USING btree (user_id);

CREATE INDEX idx_users_email ON public.users USING btree (email);

CREATE INDEX idx_users_is_active ON public.users USING btree (is_active);

CREATE INDEX idx_users_role ON public.users USING btree (role);

CREATE UNIQUE INDEX imported_keywords_pkey ON public.imported_keywords USING btree (id);

CREATE UNIQUE INDEX indexed_documents_pkey ON public.indexed_documents USING btree (id);

CREATE UNIQUE INDEX "infographicsDetails_pkey" ON public."infographicDetails" USING btree (id);

CREATE UNIQUE INDEX keyword_intelligence_pkey ON public.keyword_intelligence USING btree (id);

CREATE UNIQUE INDEX keyword_opportunities_reports_pkey ON public.keyword_opportunities_reports USING btree (id);

CREATE UNIQUE INDEX keyword_research_sessions_pkey ON public.keyword_research_sessions USING btree (id);

CREATE UNIQUE INDEX keywords_pkey ON public.keywords USING btree (id);

CREATE UNIQUE INDEX lindex_collections_pkey ON public.lindex_collections USING btree (id);

CREATE INDEX lindex_embedding_chunk_embedding_idx ON public.lindex_embedding_chunk USING ivfflat (vec extensions.vector_cosine_ops) WITH (lists='100');

CREATE UNIQUE INDEX lindex_embedding_chunk_pkey ON public.lindex_embedding_chunk USING btree (id);

CREATE UNIQUE INDEX llm_configurations_pkey ON public.llm_configurations USING btree (id);

CREATE UNIQUE INDEX llm_providers_pkey ON public.llm_providers USING btree (id);

CREATE UNIQUE INDEX manual_action_suggestions_pkey ON public.manual_action_suggestions USING btree (id);

CREATE UNIQUE INDEX offer_analytics_pkey ON public.offer_analytics USING btree (id);

CREATE UNIQUE INDEX offer_research_sessions_pkey ON public.offer_research_sessions USING btree (id);

CREATE UNIQUE INDEX "postTypes_pkey" ON public."postTypes" USING btree (id);

CREATE UNIQUE INDEX "promptDetailsByType_pkey" ON public."sectionSpecificPrompts" USING btree (id);

CREATE UNIQUE INDEX research_program_links_pkey ON public.research_program_links USING btree (id);

CREATE UNIQUE INDEX research_program_links_research_session_id_program_id_key ON public.research_program_links USING btree (research_session_id, program_id);

CREATE UNIQUE INDEX research_topics_pkey ON public.research_topics USING btree (id);

CREATE UNIQUE INDEX seasonal_calendar_pkey ON public.seasonal_calendar USING btree (id);

CREATE UNIQUE INDEX test_pkey ON public."PostLinks" USING btree (id);

CREATE UNIQUE INDEX topic_decompositions_pkey ON public.topic_decompositions USING btree (id);

CREATE UNIQUE INDEX trend_analyses_pkey ON public.trend_analyses USING btree (id);

CREATE UNIQUE INDEX trend_analysis_pkey ON public.trend_analysis USING btree (id);

CREATE UNIQUE INDEX trend_predictions_pkey ON public.trend_predictions USING btree (id);

CREATE UNIQUE INDEX trending_topics_pkey ON public.trending_topics USING btree (id);

CREATE UNIQUE INDEX user_offer_preferences_pkey ON public.user_offer_preferences USING btree (id);

CREATE UNIQUE INDEX user_profile_pkey ON public.user_profile USING btree (id);

CREATE UNIQUE INDEX users_email_key ON public.users USING btree (email);

CREATE UNIQUE INDEX users_pkey ON public.users USING btree (id);

alter table "public"."PlannedArticles" add constraint "PlannedArticles_pkey" PRIMARY KEY using index "PlannedArticles_pkey";

alter table "public"."PostLinks" add constraint "test_pkey" PRIMARY KEY using index "test_pkey";

alter table "public"."RSS" add constraint "RSS_pkey" PRIMARY KEY using index "RSS_pkey";

alter table "public"."TableOfContents" add constraint "TableOfContents_pkey" PRIMARY KEY using index "TableOfContents_pkey";

alter table "public"."Titles" add constraint "Titles_pkey" PRIMARY KEY using index "Titles_pkey";

alter table "public"."Titles_citations" add constraint "Titles_citations_pkey" PRIMARY KEY using index "Titles_citations_pkey";

alter table "public"."affiliate_offers" add constraint "affiliate_offers_pkey" PRIMARY KEY using index "affiliate_offers_pkey";

alter table "public"."affiliate_programs" add constraint "affiliate_programs_pkey" PRIMARY KEY using index "affiliate_programs_pkey";

alter table "public"."affiliate_research" add constraint "affiliate_research_pkey" PRIMARY KEY using index "affiliate_research_pkey";

alter table "public"."application_settings" add constraint "application_settings_pkey" PRIMARY KEY using index "application_settings_pkey";

alter table "public"."blog_generation_results" add constraint "blog_generation_results_pkey" PRIMARY KEY using index "blog_generation_results_pkey";

alter table "public"."blog_idea_keyword_assignments" add constraint "blog_idea_keyword_assignments_pkey" PRIMARY KEY using index "blog_idea_keyword_assignments_pkey";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_pkey" PRIMARY KEY using index "blog_idea_performance_pkey";

alter table "public"."blog_idea_templates" add constraint "blog_idea_templates_pkey" PRIMARY KEY using index "blog_idea_templates_pkey";

alter table "public"."blog_ideas" add constraint "blog_ideas_pkey" PRIMARY KEY using index "blog_ideas_pkey";

alter table "public"."categoriesByPost" add constraint "CategoriesByPost_pkey" PRIMARY KEY using index "CategoriesByPost_pkey";

alter table "public"."competitive_intelligence" add constraint "competitive_intelligence_pkey" PRIMARY KEY using index "competitive_intelligence_pkey";

alter table "public"."content_calendar" add constraint "content_calendar_pkey" PRIMARY KEY using index "content_calendar_pkey";

alter table "public"."content_ideas" add constraint "content_ideas_pkey" PRIMARY KEY using index "content_ideas_pkey";

alter table "public"."content_opportunities" add constraint "content_opportunities_pkey" PRIMARY KEY using index "content_opportunities_pkey";

alter table "public"."embeddings" add constraint "embeddings_pkey" PRIMARY KEY using index "embeddings_pkey";

alter table "public"."geographic_insights" add constraint "geographic_insights_pkey" PRIMARY KEY using index "geographic_insights_pkey";

alter table "public"."imported_keywords" add constraint "imported_keywords_pkey" PRIMARY KEY using index "imported_keywords_pkey";

alter table "public"."indexed_documents" add constraint "indexed_documents_pkey" PRIMARY KEY using index "indexed_documents_pkey";

alter table "public"."infographic" add constraint "Infographics_pkey" PRIMARY KEY using index "Infographics_pkey";

alter table "public"."infographicDetails" add constraint "infographicsDetails_pkey" PRIMARY KEY using index "infographicsDetails_pkey";

alter table "public"."keyword_intelligence" add constraint "keyword_intelligence_pkey" PRIMARY KEY using index "keyword_intelligence_pkey";

alter table "public"."keyword_opportunities_reports" add constraint "keyword_opportunities_reports_pkey" PRIMARY KEY using index "keyword_opportunities_reports_pkey";

alter table "public"."keyword_research_sessions" add constraint "keyword_research_sessions_pkey" PRIMARY KEY using index "keyword_research_sessions_pkey";

alter table "public"."keywords" add constraint "keywords_pkey" PRIMARY KEY using index "keywords_pkey";

alter table "public"."lindex_collections" add constraint "lindex_collections_pkey" PRIMARY KEY using index "lindex_collections_pkey";

alter table "public"."lindex_documents" add constraint "Books_pkey" PRIMARY KEY using index "Books_pkey";

alter table "public"."lindex_embedding_chunk" add constraint "lindex_embedding_chunk_pkey" PRIMARY KEY using index "lindex_embedding_chunk_pkey";

alter table "public"."lindex_sections" add constraint "documents_pkey" PRIMARY KEY using index "documents_pkey";

alter table "public"."llm_configurations" add constraint "llm_configurations_pkey" PRIMARY KEY using index "llm_configurations_pkey";

alter table "public"."llm_providers" add constraint "llm_providers_pkey" PRIMARY KEY using index "llm_providers_pkey";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_pkey" PRIMARY KEY using index "manual_action_suggestions_pkey";

alter table "public"."mySources" add constraint "KnowledgeBase_pkey" PRIMARY KEY using index "KnowledgeBase_pkey";

alter table "public"."offer_analytics" add constraint "offer_analytics_pkey" PRIMARY KEY using index "offer_analytics_pkey";

alter table "public"."offer_research_sessions" add constraint "offer_research_sessions_pkey" PRIMARY KEY using index "offer_research_sessions_pkey";

alter table "public"."postTypes" add constraint "postTypes_pkey" PRIMARY KEY using index "postTypes_pkey";

alter table "public"."research_program_links" add constraint "research_program_links_pkey" PRIMARY KEY using index "research_program_links_pkey";

alter table "public"."research_topics" add constraint "research_topics_pkey" PRIMARY KEY using index "research_topics_pkey";

alter table "public"."seasonal_calendar" add constraint "seasonal_calendar_pkey" PRIMARY KEY using index "seasonal_calendar_pkey";

alter table "public"."sectionSpecificPrompts" add constraint "promptDetailsByType_pkey" PRIMARY KEY using index "promptDetailsByType_pkey";

alter table "public"."summaries" add constraint "Summaries_pkey" PRIMARY KEY using index "Summaries_pkey";

alter table "public"."topic_decompositions" add constraint "topic_decompositions_pkey" PRIMARY KEY using index "topic_decompositions_pkey";

alter table "public"."trend_analyses" add constraint "trend_analyses_pkey" PRIMARY KEY using index "trend_analyses_pkey";

alter table "public"."trend_analysis" add constraint "trend_analysis_pkey" PRIMARY KEY using index "trend_analysis_pkey";

alter table "public"."trend_predictions" add constraint "trend_predictions_pkey" PRIMARY KEY using index "trend_predictions_pkey";

alter table "public"."trending_topics" add constraint "trending_topics_pkey" PRIMARY KEY using index "trending_topics_pkey";

alter table "public"."user_offer_preferences" add constraint "user_offer_preferences_pkey" PRIMARY KEY using index "user_offer_preferences_pkey";

alter table "public"."user_profile" add constraint "user_profile_pkey" PRIMARY KEY using index "user_profile_pkey";

alter table "public"."users" add constraint "users_pkey" PRIMARY KEY using index "users_pkey";

alter table "public"."wordPress_details" add constraint "WordPress_details_pkey" PRIMARY KEY using index "WordPress_details_pkey";

alter table "public"."TableOfContents" add constraint "TableOfContents_user_id_fkey" FOREIGN KEY (user_id) REFERENCES user_profile(id) not valid;

alter table "public"."TableOfContents" validate constraint "TableOfContents_user_id_fkey";

alter table "public"."Titles" add constraint "Titles_audience_alignment_score_check" CHECK (((audience_alignment_score >= 0) AND (audience_alignment_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_audience_alignment_score_check";

alter table "public"."Titles" add constraint "Titles_business_impact_score_check" CHECK (((business_impact_score >= 0) AND (business_impact_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_business_impact_score_check";

alter table "public"."Titles" add constraint "Titles_competition_score_check" CHECK (((competition_score >= 0) AND (competition_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_competition_score_check";

alter table "public"."Titles" add constraint "Titles_content_feasibility_score_check" CHECK (((content_feasibility_score >= 0) AND (content_feasibility_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_content_feasibility_score_check";

alter table "public"."Titles" add constraint "Titles_overall_quality_score_check" CHECK (((overall_quality_score >= 0) AND (overall_quality_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_overall_quality_score_check";

alter table "public"."Titles" add constraint "Titles_priority_level_check" CHECK ((priority_level = ANY (ARRAY['high'::text, 'medium'::text, 'low'::text]))) not valid;

alter table "public"."Titles" validate constraint "Titles_priority_level_check";

alter table "public"."Titles" add constraint "Titles_seo_optimization_score_check" CHECK (((seo_optimization_score >= 0) AND (seo_optimization_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_seo_optimization_score_check";

alter table "public"."Titles" add constraint "Titles_traffic_potential_score_check" CHECK (((traffic_potential_score >= 0) AND (traffic_potential_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_traffic_potential_score_check";

alter table "public"."Titles" add constraint "Titles_user_id_fkey" FOREIGN KEY (user_id) REFERENCES user_profile(id) not valid;

alter table "public"."Titles" validate constraint "Titles_user_id_fkey";

alter table "public"."Titles" add constraint "Titles_viral_potential_score_check" CHECK (((viral_potential_score >= 0) AND (viral_potential_score <= 100))) not valid;

alter table "public"."Titles" validate constraint "Titles_viral_potential_score_check";

alter table "public"."Titles" add constraint "Titles_workflow_status_check" CHECK ((workflow_status = ANY (ARRAY['idea_selected'::text, 'content_planned'::text, 'writing_in_progress'::text, 'ready_for_review'::text, 'ready_to_publish'::text, 'published'::text, 'needs_revision'::text]))) not valid;

alter table "public"."Titles" validate constraint "Titles_workflow_status_check";

alter table "public"."Titles" add constraint "Titles_wp_status_check" CHECK ((wp_status = ANY (ARRAY['draft'::text, 'pending'::text, 'publish'::text, 'private'::text]))) not valid;

alter table "public"."Titles" validate constraint "Titles_wp_status_check";

alter table "public"."affiliate_programs" add constraint "affiliate_programs_status_check" CHECK (((status)::text = ANY (ARRAY[('active'::character varying)::text, ('inactive'::character varying)::text, ('suspended'::character varying)::text, ('closed'::character varying)::text]))) not valid;

alter table "public"."affiliate_programs" validate constraint "affiliate_programs_status_check";

alter table "public"."affiliate_programs" add constraint "affiliate_programs_verification_status_check" CHECK (((verification_status)::text = ANY (ARRAY[('verified'::character varying)::text, ('unverified'::character varying)::text, ('pending'::character varying)::text, ('failed'::character varying)::text]))) not valid;

alter table "public"."affiliate_programs" validate constraint "affiliate_programs_verification_status_check";

alter table "public"."affiliate_research" add constraint "affiliate_research_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE not valid;

alter table "public"."affiliate_research" validate constraint "affiliate_research_user_id_fkey";

alter table "public"."api_keys" add constraint "api_keys_key_name_key" UNIQUE using index "api_keys_key_name_key";

alter table "public"."blog_generation_results" add constraint "blog_generation_results_average_quality_score_check" CHECK (((average_quality_score >= (0)::numeric) AND (average_quality_score <= (100)::numeric))) not valid;

alter table "public"."blog_generation_results" validate constraint "blog_generation_results_average_quality_score_check";

alter table "public"."blog_generation_results" add constraint "blog_generation_results_processing_time_seconds_check" CHECK ((processing_time_seconds >= (0)::numeric)) not valid;

alter table "public"."blog_generation_results" validate constraint "blog_generation_results_processing_time_seconds_check";

alter table "public"."blog_generation_results" add constraint "blog_generation_results_total_ideas_generated_check" CHECK ((total_ideas_generated >= 0)) not valid;

alter table "public"."blog_generation_results" validate constraint "blog_generation_results_total_ideas_generated_check";

alter table "public"."blog_generation_results" add constraint "blog_generation_results_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."blog_generation_results" validate constraint "blog_generation_results_trend_analysis_id_fkey";

alter table "public"."blog_idea_keyword_assignments" add constraint "blog_idea_keyword_assignments_blog_idea_id_fkey" FOREIGN KEY (blog_idea_id) REFERENCES blog_ideas(id) ON DELETE CASCADE not valid;

alter table "public"."blog_idea_keyword_assignments" validate constraint "blog_idea_keyword_assignments_blog_idea_id_fkey";

alter table "public"."blog_idea_keyword_assignments" add constraint "blog_idea_keyword_assignments_blog_idea_id_imported_keyword_key" UNIQUE using index "blog_idea_keyword_assignments_blog_idea_id_imported_keyword_key";

alter table "public"."blog_idea_keyword_assignments" add constraint "blog_idea_keyword_assignments_imported_keyword_id_fkey" FOREIGN KEY (imported_keyword_id) REFERENCES imported_keywords(id) ON DELETE CASCADE not valid;

alter table "public"."blog_idea_keyword_assignments" validate constraint "blog_idea_keyword_assignments_imported_keyword_id_fkey";

alter table "public"."blog_idea_keyword_assignments" add constraint "blog_idea_keyword_assignments_keyword_type_check" CHECK ((keyword_type = ANY (ARRAY['primary'::text, 'secondary'::text, 'suggested'::text]))) not valid;

alter table "public"."blog_idea_keyword_assignments" validate constraint "blog_idea_keyword_assignments_keyword_type_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_actual_word_count_check" CHECK ((actual_word_count > 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_actual_word_count_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_average_search_position_check" CHECK ((average_search_position > (0)::numeric)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_average_search_position_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_backlinks_count_check" CHECK ((backlinks_count >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_backlinks_count_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_blog_idea_id_fkey" FOREIGN KEY (blog_idea_id) REFERENCES blog_ideas(id) ON DELETE CASCADE not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_blog_idea_id_fkey";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_bounce_rate_check" CHECK (((bounce_rate >= (0)::numeric) AND (bounce_rate <= (100)::numeric))) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_bounce_rate_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_email_signups_check" CHECK ((email_signups >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_email_signups_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_lead_conversions_check" CHECK ((lead_conversions >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_lead_conversions_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_organic_traffic_percentage_check" CHECK (((organic_traffic_percentage >= (0)::numeric) AND (organic_traffic_percentage <= (100)::numeric))) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_organic_traffic_percentage_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_page_views_check" CHECK ((page_views >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_page_views_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_prediction_accuracy_score_check" CHECK (((prediction_accuracy_score >= 0) AND (prediction_accuracy_score <= 100))) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_prediction_accuracy_score_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_revenue_attributed_check" CHECK ((revenue_attributed >= (0)::numeric)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_revenue_attributed_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_social_shares_check" CHECK ((social_shares >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_social_shares_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_time_on_page_seconds_check" CHECK ((time_on_page_seconds >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_time_on_page_seconds_check";

alter table "public"."blog_idea_performance" add constraint "blog_idea_performance_unique_visitors_check" CHECK ((unique_visitors >= 0)) not valid;

alter table "public"."blog_idea_performance" validate constraint "blog_idea_performance_unique_visitors_check";

alter table "public"."blog_idea_templates" add constraint "blog_idea_templates_content_format_check" CHECK ((content_format = ANY (ARRAY['how_to_guide'::text, 'listicle'::text, 'case_study'::text, 'comparison'::text, 'trend_analysis'::text, 'tutorial'::text, 'review'::text, 'interview'::text, 'opinion'::text, 'news_analysis'::text, 'resource_roundup'::text, 'checklist'::text, 'template'::text, 'interactive_tool'::text, 'infographic'::text]))) not valid;

alter table "public"."blog_idea_templates" validate constraint "blog_idea_templates_content_format_check";

alter table "public"."blog_ideas" add constraint "blog_ideas_source_opportunity_id_fkey" FOREIGN KEY (source_opportunity_id) REFERENCES content_opportunities(id) not valid;

alter table "public"."blog_ideas" validate constraint "blog_ideas_source_opportunity_id_fkey";

alter table "public"."blog_ideas" add constraint "blog_ideas_source_topic_id_fkey" FOREIGN KEY (source_topic_id) REFERENCES trending_topics(id) not valid;

alter table "public"."blog_ideas" validate constraint "blog_ideas_source_topic_id_fkey";

alter table "public"."blog_ideas" add constraint "blog_ideas_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) not valid;

alter table "public"."blog_ideas" validate constraint "blog_ideas_trend_analysis_id_fkey";

alter table "public"."blog_ideas" add constraint "blog_ideas_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."blog_ideas" validate constraint "blog_ideas_user_id_fkey";

alter table "public"."blog_ideas" add constraint "valid_content_format" CHECK ((content_format = ANY (ARRAY['how_to_guide'::text, 'listicle'::text, 'case_study'::text, 'comparison'::text, 'trend_analysis'::text, 'tutorial'::text, 'review'::text, 'interview'::text, 'opinion'::text, 'news_analysis'::text, 'resource_roundup'::text, 'checklist'::text, 'template'::text, 'interactive_tool'::text, 'infographic'::text]))) not valid;

alter table "public"."blog_ideas" validate constraint "valid_content_format";

alter table "public"."blog_ideas" add constraint "valid_difficulty" CHECK ((difficulty_level = ANY (ARRAY['beginner'::text, 'intermediate'::text, 'advanced'::text, 'expert'::text]))) not valid;

alter table "public"."blog_ideas" validate constraint "valid_difficulty";

alter table "public"."blog_ideas" add constraint "valid_priority" CHECK ((priority_level = ANY (ARRAY['high'::text, 'medium'::text, 'low'::text]))) not valid;

alter table "public"."blog_ideas" validate constraint "valid_priority";

alter table "public"."blog_ideas" add constraint "valid_scores" CHECK (((overall_quality_score >= 0) AND (overall_quality_score <= 100) AND (viral_potential_score >= 0) AND (viral_potential_score <= 100) AND (seo_optimization_score >= 0) AND (seo_optimization_score <= 100) AND (audience_alignment_score >= 0) AND (audience_alignment_score <= 100) AND (content_feasibility_score >= 0) AND (content_feasibility_score <= 100) AND (business_impact_score >= 0) AND (business_impact_score <= 100))) not valid;

alter table "public"."blog_ideas" validate constraint "valid_scores";

alter table "public"."categoriesByPost" add constraint "CategoriesByPost_user_id_fkey" FOREIGN KEY (user_id) REFERENCES user_profile(id) not valid;

alter table "public"."categoriesByPost" validate constraint "CategoriesByPost_user_id_fkey";

alter table "public"."competitive_intelligence" add constraint "competitive_intelligence_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."competitive_intelligence" validate constraint "competitive_intelligence_trend_analysis_id_fkey";

alter table "public"."content_calendar" add constraint "content_calendar_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) not valid;

alter table "public"."content_calendar" validate constraint "content_calendar_trend_analysis_id_fkey";

alter table "public"."content_calendar" add constraint "content_calendar_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."content_calendar" validate constraint "content_calendar_user_id_fkey";

alter table "public"."content_ideas" add constraint "content_ideas_content_type_check" CHECK (((content_type)::text = ANY (ARRAY[('blog'::character varying)::text, ('software'::character varying)::text]))) not valid;

alter table "public"."content_ideas" validate constraint "content_ideas_content_type_check";

alter table "public"."content_ideas" add constraint "content_ideas_priority_check" CHECK (((priority)::text = ANY ((ARRAY['low'::character varying, 'medium'::character varying, 'high'::character varying, 'urgent'::character varying])::text[]))) not valid;

alter table "public"."content_ideas" validate constraint "content_ideas_priority_check";

alter table "public"."content_ideas" add constraint "content_ideas_status_check" CHECK (((status)::text = ANY ((ARRAY['draft'::character varying, 'in_progress'::character varying, 'completed'::character varying, 'published'::character varying, 'archived'::character varying])::text[]))) not valid;

alter table "public"."content_ideas" validate constraint "content_ideas_status_check";

alter table "public"."content_opportunities" add constraint "content_opportunities_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."content_opportunities" validate constraint "content_opportunities_trend_analysis_id_fkey";

alter table "public"."content_opportunities" add constraint "content_opportunities_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."content_opportunities" validate constraint "content_opportunities_user_id_fkey";

alter table "public"."geographic_insights" add constraint "geographic_insights_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."geographic_insights" validate constraint "geographic_insights_trend_analysis_id_fkey";

alter table "public"."imported_keywords" add constraint "imported_keywords_competition_check" CHECK ((competition = ANY (ARRAY['low'::text, 'medium'::text, 'high'::text]))) not valid;

alter table "public"."imported_keywords" validate constraint "imported_keywords_competition_check";

alter table "public"."imported_keywords" add constraint "imported_keywords_search_intent_check" CHECK ((search_intent = ANY (ARRAY['informational'::text, 'commercial'::text, 'navigational'::text, 'transactional'::text]))) not valid;

alter table "public"."imported_keywords" validate constraint "imported_keywords_search_intent_check";

alter table "public"."imported_keywords" add constraint "imported_keywords_session_id_fkey" FOREIGN KEY (session_id) REFERENCES keyword_research_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."imported_keywords" validate constraint "imported_keywords_session_id_fkey";

alter table "public"."imported_keywords" add constraint "imported_keywords_trend_check" CHECK ((trend = ANY (ARRAY['rising'::text, 'stable'::text, 'declining'::text]))) not valid;

alter table "public"."imported_keywords" validate constraint "imported_keywords_trend_check";

alter table "public"."keyword_intelligence" add constraint "keyword_intelligence_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."keyword_intelligence" validate constraint "keyword_intelligence_trend_analysis_id_fkey";

alter table "public"."keyword_intelligence" add constraint "keyword_intelligence_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."keyword_intelligence" validate constraint "keyword_intelligence_user_id_fkey";

alter table "public"."keyword_opportunities_reports" add constraint "keyword_opportunities_reports_analysis_id_fkey" FOREIGN KEY (analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."keyword_opportunities_reports" validate constraint "keyword_opportunities_reports_analysis_id_fkey";

alter table "public"."keyword_opportunities_reports" add constraint "keyword_opportunities_reports_session_id_fkey" FOREIGN KEY (session_id) REFERENCES keyword_research_sessions(id) ON DELETE CASCADE not valid;

alter table "public"."keyword_opportunities_reports" validate constraint "keyword_opportunities_reports_session_id_fkey";

alter table "public"."keyword_research_sessions" add constraint "keyword_research_sessions_source_tool_check" CHECK ((source_tool = ANY (ARRAY['ahrefs'::text, 'semrush'::text, 'moz'::text, 'ubersuggest'::text, 'kwfinder'::text, 'custom'::text]))) not valid;

alter table "public"."keyword_research_sessions" validate constraint "keyword_research_sessions_source_tool_check";

alter table "public"."keyword_research_sessions" add constraint "keyword_research_sessions_validation_status_check" CHECK ((validation_status = ANY (ARRAY['pending'::text, 'passed'::text, 'failed'::text]))) not valid;

alter table "public"."keyword_research_sessions" validate constraint "keyword_research_sessions_validation_status_check";

alter table "public"."keywords" add constraint "keywords_source_check" CHECK (((source)::text = ANY (ARRAY[('llm'::character varying)::text, ('ahrefs'::character varying)::text, ('manual'::character varying)::text]))) not valid;

alter table "public"."keywords" validate constraint "keywords_source_check";

alter table "public"."llm_configurations" add constraint "llm_configurations_provider_id_fkey" FOREIGN KEY (provider_id) REFERENCES llm_providers(id) ON DELETE CASCADE not valid;

alter table "public"."llm_configurations" validate constraint "llm_configurations_provider_id_fkey";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_action_type_check" CHECK ((action_type = ANY (ARRAY['textbook'::text, 'dataset'::text, 'tool_development'::text, 'expert_interview'::text, 'course_creation'::text, 'template_creation'::text, 'partnership'::text, 'research_study'::text]))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_action_type_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_difficulty_level_check" CHECK ((difficulty_level = ANY (ARRAY['beginner'::text, 'intermediate'::text, 'advanced'::text]))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_difficulty_level_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_feasibility_score_check" CHECK (((feasibility_score >= 0) AND (feasibility_score <= 100))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_feasibility_score_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_impact_score_check" CHECK (((impact_score >= 0) AND (impact_score <= 100))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_impact_score_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_priority_level_check" CHECK ((priority_level = ANY (ARRAY['low'::text, 'medium'::text, 'high'::text, 'critical'::text]))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_priority_level_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_priority_score_check" CHECK (((priority_score >= (0)::numeric) AND (priority_score <= (100)::numeric))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_priority_score_check";

alter table "public"."manual_action_suggestions" add constraint "manual_action_suggestions_status_check" CHECK ((status = ANY (ARRAY['suggested'::text, 'in_progress'::text, 'completed'::text, 'rejected'::text]))) not valid;

alter table "public"."manual_action_suggestions" validate constraint "manual_action_suggestions_status_check";

alter table "public"."offer_analytics" add constraint "offer_analytics_offer_id_fkey" FOREIGN KEY (offer_id) REFERENCES affiliate_offers(id) ON DELETE CASCADE not valid;

alter table "public"."offer_analytics" validate constraint "offer_analytics_offer_id_fkey";

alter table "public"."offer_analytics" add constraint "offer_analytics_research_session_id_fkey" FOREIGN KEY (research_session_id) REFERENCES offer_research_sessions(id) ON DELETE SET NULL not valid;

alter table "public"."offer_analytics" validate constraint "offer_analytics_research_session_id_fkey";

alter table "public"."offer_analytics" add constraint "offer_analytics_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE not valid;

alter table "public"."offer_analytics" validate constraint "offer_analytics_user_id_fkey";

alter table "public"."offer_research_sessions" add constraint "offer_research_sessions_status_check" CHECK (((status)::text = ANY (ARRAY[('active'::character varying)::text, ('completed'::character varying)::text, ('archived'::character varying)::text]))) not valid;

alter table "public"."offer_research_sessions" validate constraint "offer_research_sessions_status_check";

alter table "public"."offer_research_sessions" add constraint "offer_research_sessions_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE not valid;

alter table "public"."offer_research_sessions" validate constraint "offer_research_sessions_user_id_fkey";

alter table "public"."research_program_links" add constraint "research_program_links_link_type_check" CHECK ((link_type = ANY (ARRAY['existing'::text, 'new'::text, 'reused'::text]))) not valid;

alter table "public"."research_program_links" validate constraint "research_program_links_link_type_check";

alter table "public"."research_program_links" add constraint "research_program_links_research_session_id_program_id_key" UNIQUE using index "research_program_links_research_session_id_program_id_key";

alter table "public"."research_topics" add constraint "research_topics_status_check" CHECK (((status)::text = ANY (ARRAY[('active'::character varying)::text, ('completed'::character varying)::text, ('archived'::character varying)::text]))) not valid;

alter table "public"."research_topics" validate constraint "research_topics_status_check";

alter table "public"."research_topics" add constraint "research_topics_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE not valid;

alter table "public"."research_topics" validate constraint "research_topics_user_id_fkey";

alter table "public"."research_topics" add constraint "research_topics_version_check" CHECK ((version > 0)) not valid;

alter table "public"."research_topics" validate constraint "research_topics_version_check";

alter table "public"."seasonal_calendar" add constraint "seasonal_calendar_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."seasonal_calendar" validate constraint "seasonal_calendar_trend_analysis_id_fkey";

alter table "public"."topic_decompositions" add constraint "topic_decompositions_research_topic_id_fkey" FOREIGN KEY (research_topic_id) REFERENCES research_topics(id) ON DELETE CASCADE not valid;

alter table "public"."topic_decompositions" validate constraint "topic_decompositions_research_topic_id_fkey";

alter table "public"."trend_analyses" add constraint "trend_analyses_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."trend_analyses" validate constraint "trend_analyses_user_id_fkey";

alter table "public"."trend_analysis" add constraint "trend_analysis_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE not valid;

alter table "public"."trend_analysis" validate constraint "trend_analysis_user_id_fkey";

alter table "public"."trend_predictions" add constraint "trend_predictions_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."trend_predictions" validate constraint "trend_predictions_trend_analysis_id_fkey";

alter table "public"."trending_topics" add constraint "trending_topics_trend_analysis_id_fkey" FOREIGN KEY (trend_analysis_id) REFERENCES trend_analyses(id) ON DELETE CASCADE not valid;

alter table "public"."trending_topics" validate constraint "trending_topics_trend_analysis_id_fkey";

alter table "public"."trending_topics" add constraint "trending_topics_user_id_fkey" FOREIGN KEY (user_id) REFERENCES auth.users(id) not valid;

alter table "public"."trending_topics" validate constraint "trending_topics_user_id_fkey";

alter table "public"."user_offer_preferences" add constraint "user_offer_preferences_user_id_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE not valid;

alter table "public"."user_offer_preferences" validate constraint "user_offer_preferences_user_id_fkey";

alter table "public"."user_profile" add constraint "user_profile_id_fkey" FOREIGN KEY (id) REFERENCES auth.users(id) not valid;

alter table "public"."user_profile" validate constraint "user_profile_id_fkey";

alter table "public"."users" add constraint "users_email_key" UNIQUE using index "users_email_key";

alter table "public"."users" add constraint "users_role_check" CHECK (((role)::text = ANY (ARRAY[('user'::character varying)::text, ('admin'::character varying)::text, ('moderator'::character varying)::text]))) not valid;

alter table "public"."users" validate constraint "users_role_check";

alter table "public"."wordPress_details" add constraint "wordPress_details_user_id_fkey" FOREIGN KEY (user_id) REFERENCES user_profile(id) not valid;

alter table "public"."wordPress_details" validate constraint "wordPress_details_user_id_fkey";

set check_function_bodies = off;

create or replace view "public"."blog_ideas_summary" as  SELECT bi.id,
    bi.user_id,
    bi.trend_analysis_id,
    bi.title,
    bi.content_format,
    bi.difficulty_level,
    bi.overall_quality_score,
    bi.viral_potential_score,
    bi.seo_optimization_score,
    bi.selected,
    bi.priority_level,
    bi.scheduled_publish_date,
    bi.created_at,
    ta.topic,
    ta.target_audience,
    ta.focus_area,
        CASE
            WHEN (bi.overall_quality_score >= 85) THEN 'Excellent'::text
            WHEN (bi.overall_quality_score >= 75) THEN 'High Quality'::text
            WHEN (bi.overall_quality_score >= 65) THEN 'Good'::text
            WHEN (bi.overall_quality_score >= 55) THEN 'Decent'::text
            ELSE 'Needs Work'::text
        END AS quality_tier,
    row_number() OVER (PARTITION BY bi.user_id, bi.trend_analysis_id ORDER BY bi.overall_quality_score DESC, bi.viral_potential_score DESC) AS quality_rank
   FROM (blog_ideas bi
     JOIN trend_analyses ta ON ((bi.trend_analysis_id = ta.id)));


create or replace view "public"."blog_ideas_with_keywords" as  SELECT bi.id,
    bi.user_id,
    bi.trend_analysis_id,
    bi.title,
    bi.content_format,
    bi.overall_quality_score,
    bi.seo_optimization_score,
    bi.traffic_potential_score,
    bi.competition_score,
    bi.keyword_research_enhanced,
    bi.enhanced_primary_keywords,
    bi.enhanced_secondary_keywords,
    ((bi.keyword_research_data ->> 'total_search_volume'::text))::integer AS total_search_volume,
    ((bi.keyword_research_data ->> 'average_difficulty'::text))::numeric AS average_difficulty,
    ((bi.keyword_research_data ->> 'average_cpc'::text))::numeric AS average_cpc,
    jsonb_array_length(COALESCE(bi.enhanced_primary_keywords, '[]'::jsonb)) AS primary_keywords_count,
    jsonb_array_length(COALESCE(bi.enhanced_secondary_keywords, '[]'::jsonb)) AS secondary_keywords_count,
    bi.enhancement_timestamp,
    bi.created_at,
    bi.updated_at
   FROM blog_ideas bi
  WHERE (bi.keyword_research_enhanced = true);


CREATE OR REPLACE FUNCTION public.calculate_keyword_opportunity_score(p_search_volume integer, p_difficulty numeric, p_cpc numeric, p_intent text)
 RETURNS integer
 LANGUAGE plpgsql
AS $function$
DECLARE
    opportunity_score INTEGER := 0;
BEGIN
    -- Search volume factor (0-40 points)
    IF p_search_volume >= 1000 THEN
        opportunity_score := opportunity_score + 40;
    ELSIF p_search_volume >= 500 THEN
        opportunity_score := opportunity_score + 30;
    ELSIF p_search_volume >= 200 THEN
        opportunity_score := opportunity_score + 20;
    ELSIF p_search_volume >= 100 THEN
        opportunity_score := opportunity_score + 10;
    END IF;
    
    -- Difficulty factor (0-30 points, inverse)
    opportunity_score := opportunity_score + GREATEST(0, 30 - p_difficulty::INTEGER);
    
    -- CPC factor (0-20 points)
    IF p_cpc >= 3.0 THEN
        opportunity_score := opportunity_score + 20;
    ELSIF p_cpc >= 2.0 THEN
        opportunity_score := opportunity_score + 15;
    ELSIF p_cpc >= 1.0 THEN
        opportunity_score := opportunity_score + 10;
    ELSIF p_cpc >= 0.5 THEN
        opportunity_score := opportunity_score + 5;
    END IF;
    
    -- Intent factor (0-10 points)
    CASE p_intent
        WHEN 'informational' THEN opportunity_score := opportunity_score + 10;
        WHEN 'commercial' THEN opportunity_score := opportunity_score + 8;
        WHEN 'transactional' THEN opportunity_score := opportunity_score + 6;
        ELSE opportunity_score := opportunity_score + 5;
    END CASE;
    
    RETURN LEAST(100, opportunity_score);
END;
$function$
;

CREATE OR REPLACE FUNCTION public.cleanup_old_generation_results()
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
    DELETE FROM public.blog_generation_results
    WHERE id NOT IN (
        SELECT id FROM (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY user_id ORDER BY created_at DESC
            ) as rn
            FROM public.blog_generation_results
        ) ranked
        WHERE rn <= 50
    );
END;
$function$
;

create or replace view "public"."content_calendar_summary" as  SELECT cc.id,
    cc.user_id,
    cc.trend_analysis_id,
    cc.publishing_strategy,
    cc.priority_scheduling,
    cc.seasonal_optimization,
    cc.content_series_opportunities,
    cc.format_distribution,
    cc.estimated_resource_requirements,
    cc.created_at,
    cc.updated_at,
    ta.topic,
    ta.target_audience,
    count(bi.id) AS total_blog_ideas,
    count(
        CASE
            WHEN (bi.selected = true) THEN 1
            ELSE NULL::integer
        END) AS selected_blog_ideas,
    avg(bi.overall_quality_score) AS average_quality_score,
    max(bi.overall_quality_score) AS highest_quality_score
   FROM ((content_calendar cc
     JOIN trend_analyses ta ON ((cc.trend_analysis_id = ta.id)))
     LEFT JOIN blog_ideas bi ON ((cc.trend_analysis_id = bi.trend_analysis_id)))
  GROUP BY cc.id, ta.topic, ta.target_audience;


CREATE OR REPLACE FUNCTION public.delete_vector_by_docid(collection_name_param text, doc_id_param text)
 RETURNS SETOF integer
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
DECLARE
  query TEXT;
  affected_rows INT;
BEGIN
  -- Build a dynamic SQL query with proper escaping using format()
  query := format('DELETE FROM vecs.%I WHERE metadata->>''docid'' = %L RETURNING 1', 
                 collection_name_param, doc_id_param);
  
  -- Execute the query and get affected rows
  EXECUTE query INTO affected_rows;
  
  -- Return the number of affected rows
  RETURN QUERY SELECT affected_rows;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.get_collection_docids(target_collection text)
 RETURNS TABLE(docid text)
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$
BEGIN
  RETURN QUERY EXECUTE format(
    'SELECT DISTINCT (metadata->>''docid'')::text 
     FROM vecs.%I 
     WHERE metadata->>''docid'' IS NOT NULL',
    target_collection
  );
END;
$function$
;

CREATE OR REPLACE FUNCTION public.get_keyword_insights_for_analysis(p_analysis_id uuid, p_user_id uuid)
 RETURNS TABLE(insight_type text, insight_title text, insight_description text, metric_value numeric, recommendation text)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    WITH keyword_stats AS (
        SELECT 
            COUNT(*) as total_keywords,
            COUNT(*) FILTER (WHERE ik.opportunity_score >= 70) as high_opportunity,
            COUNT(*) FILTER (WHERE ik.keyword_difficulty <= 30) as quick_wins,
            COUNT(*) FILTER (WHERE ik.search_intent = 'informational') as informational_count,
            SUM(ik.search_volume) as total_volume,
            AVG(ik.keyword_difficulty) as avg_difficulty
        FROM blog_idea_keyword_assignments bka
        JOIN imported_keywords ik ON bka.imported_keyword_id = ik.id
        JOIN blog_ideas bi ON bka.blog_idea_id = bi.id
        WHERE bi.trend_analysis_id = p_analysis_id 
        AND bi.user_id = p_user_id
        AND bka.is_active = true
    )
    SELECT 
        'opportunity'::TEXT,
        'High Opportunity Keywords'::TEXT,
        format('Found %s high-opportunity keywords with excellent ranking potential', ks.high_opportunity),
        ks.high_opportunity::NUMERIC,
        'Prioritize these keywords for immediate content creation'::TEXT
    FROM keyword_stats ks
    WHERE ks.high_opportunity > 0
    
    UNION ALL
    
    SELECT 
        'quick_wins'::TEXT,
        'Quick Win Opportunities'::TEXT,
        format('Found %s low-competition keywords for fast ranking', ks.quick_wins),
        ks.quick_wins::NUMERIC,
        'Target these keywords for quick SEO victories'::TEXT
    FROM keyword_stats ks
    WHERE ks.quick_wins > 0
    
    UNION ALL
    
    SELECT 
        'traffic_potential'::TEXT,
        'Traffic Potential'::TEXT,
        format('Total addressable search volume: %s monthly searches', ks.total_volume),
        ks.total_volume::NUMERIC,
        'High traffic potential - focus on content quality and optimization'::TEXT
    FROM keyword_stats ks
    WHERE ks.total_volume > 0;
    
END;
$function$
;

CREATE OR REPLACE FUNCTION public.get_unique_affiliate_programs(user_uuid uuid, search_topic character varying)
 RETURNS TABLE(program_id uuid, network character varying, program_name text, commission_rate numeric, commission_amount numeric, cookie_duration character varying, program_url text, relevance_score numeric, subtopic character varying)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT 
        ap.id,
        ap.network,
        ap.program_name,
        ap.commission_rate,
        ap.commission_amount,
        ap.cookie_duration,
        ap.program_url,
        asp.relevance_score,
        asp.subtopic
    FROM public.affiliate_programs ap
    JOIN public.affiliate_session_programs asp ON ap.id = asp.affiliate_program_id
    JOIN public.affiliate_research_sessions ars ON asp.research_session_id = ars.id
    WHERE ars.user_id = user_uuid 
    AND ars.topic = search_topic
    ORDER BY asp.relevance_score DESC, ap.commission_rate DESC;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.handle_new_user()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO ''
AS $function$
begin
  insert into public.user_profile (id, email, first_name, last_name)
  values (new.id, new.email, new.raw_user_meta_data ->> 'first_name', new.raw_user_meta_data ->> 'last_name');
  return new;
end;
$function$
;

CREATE OR REPLACE FUNCTION public.insert_sample_blog_templates()
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
    -- How-to Guide Template
    INSERT INTO public.blog_idea_templates (
        user_id, template_name, content_format, target_audience, description,
        outline_template, key_points_template, engagement_hooks_template, is_system_template
    ) VALUES (
        NULL, -- System template
        'How-to Guide Template',
        'how_to_guide',
        'professional',
        'Standard template for creating step-by-step how-to guides',
        '["Introduction and Problem Statement", "Prerequisites and Tools Needed", "Step-by-Step Instructions", "Common Pitfalls and Solutions", "Advanced Tips and Variations", "Conclusion and Next Steps"]'::jsonb,
        '["Clear problem definition", "Actionable steps", "Visual aids", "Real examples", "Troubleshooting guide"]'::jsonb,
        '["Hook with surprising statistic", "Promise specific outcome", "Address pain point immediately"]'::jsonb,
        true
    );

    -- Listicle Template
    INSERT INTO public.blog_idea_templates (
        user_id, template_name, content_format, target_audience, description,
        outline_template, key_points_template, engagement_hooks_template, is_system_template
    ) VALUES (
        NULL, -- System template
        'Listicle Template',
        'listicle',
        'professional',
        'Standard template for creating engaging list-based content',
        '["Compelling Introduction", "Item 1 with Explanation", "Item 2 with Explanation", "Item 3 with Explanation", "Additional Items", "Summary and Action Steps"]'::jsonb,
        '["Numbered items with clear benefits", "Supporting evidence", "Practical examples", "Implementation tips"]'::jsonb,
        '["Number-based headline", "Promise comprehensive coverage", "Tease valuable insights"]'::jsonb,
        true
    );
END;
$function$
;

CREATE OR REPLACE FUNCTION public.is_admin()
 RETURNS boolean
 LANGUAGE sql
 SECURITY DEFINER
AS $function$
  SELECT EXISTS (
    SELECT 1 FROM users 
    WHERE id = public.user_id() 
    AND role = 'admin' 
    AND is_active = true
  );
$function$
;

CREATE OR REPLACE FUNCTION public.is_authenticated()
 RETURNS boolean
 LANGUAGE sql
 SECURITY DEFINER
AS $function$
  SELECT public.user_id() IS NOT NULL;
$function$
;

create or replace view "public"."keyword_assignment_summary" as  SELECT bka.blog_idea_id,
    bi.title AS blog_idea_title,
    bka.user_id,
    count(*) AS total_assigned_keywords,
    count(*) FILTER (WHERE (bka.keyword_type = 'primary'::text)) AS primary_keywords,
    count(*) FILTER (WHERE (bka.keyword_type = 'secondary'::text)) AS secondary_keywords,
    count(*) FILTER (WHERE (bka.keyword_type = 'suggested'::text)) AS suggested_keywords,
    sum(ik.search_volume) AS total_search_volume,
    avg(ik.keyword_difficulty) AS avg_difficulty,
    avg(ik.cpc) AS avg_cpc,
    avg(bka.assignment_score) AS avg_assignment_score
   FROM ((blog_idea_keyword_assignments bka
     JOIN blog_ideas bi ON ((bka.blog_idea_id = bi.id)))
     JOIN imported_keywords ik ON ((bka.imported_keyword_id = ik.id)))
  WHERE (bka.is_active = true)
  GROUP BY bka.blog_idea_id, bi.title, bka.user_id;


CREATE OR REPLACE FUNCTION public.query_embeddings(inputembedding extensions.vector, match_threshold double precision)
 RETURNS TABLE(id bigint, content text, embedding extensions.vector, similarity double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.content,
    embeddings.embedding,
    1 - (embeddings.embedding <#> inputembedding) AS similarity
  FROM embeddings
  WHERE 1 - (embeddings.embedding <#> inputembedding) > match_threshold
    AND embeddings.cluster = false
  ORDER BY similarity DESC;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.query_embeddings_clusters(inputembedding extensions.vector, match_threshold double precision)
 RETURNS TABLE(id bigint, content text, embedding extensions.vector, similarity double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.content,
    embeddings.embedding,
    1 - (embeddings.embedding <#> inputembedding) AS similarity
  FROM embeddings
  WHERE (1 - (embeddings.embedding <#> inputembedding) > match_threshold)
    AND embeddings.cluster = true
  ORDER BY similarity DESC;
END;
$function$
;

CREATE OR REPLACE FUNCTION public."query_embeddings_llamaIndex"(inputembedding extensions.vector, match_threshold double precision)
 RETURNS TABLE(id bigint, content text, embedding extensions.vector, similarity double precision)
 LANGUAGE plpgsql
AS $function$BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.content,
    embeddings.vector AS embedding, -- Make sure this matches the expected return type (vector(384))
    1 - (embeddings.vector <#> inputembedding) AS similarity -- This should be of type float
  FROM embeddings
  WHERE 1 - (embeddings.vector <#> inputembedding) > match_threshold
     ORDER BY similarity DESC;
END;$function$
;

CREATE OR REPLACE FUNCTION public.trigger_update_opportunity_score()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.opportunity_score := calculate_keyword_opportunity_score(
        NEW.search_volume,
        NEW.keyword_difficulty,
        NEW.cpc,
        NEW.search_intent
    );
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.trigger_update_session_stats()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE keyword_research_sessions 
        SET 
            total_keywords_imported = (
                SELECT COUNT(*) FROM imported_keywords 
                WHERE session_id = NEW.session_id
            ),
            updated_at = NOW()
        WHERE id = NEW.session_id;
        
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE keyword_research_sessions 
        SET 
            total_keywords_imported = (
                SELECT COUNT(*) FROM imported_keywords 
                WHERE session_id = OLD.session_id
            ),
            updated_at = NOW()
        WHERE id = OLD.session_id;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_keyword_opportunity_scores(p_user_id uuid)
 RETURNS integer
 LANGUAGE plpgsql
AS $function$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    UPDATE imported_keywords 
    SET opportunity_score = calculate_keyword_opportunity_score(
        search_volume, 
        keyword_difficulty, 
        cpc, 
        search_intent
    ),
    updated_at = NOW()
    WHERE user_id = p_user_id;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_last_updated_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_quality_scores_from_performance()
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
    UPDATE public.blog_ideas bi
    SET overall_quality_score = LEAST(100, GREATEST(0, 
        bi.overall_quality_score + 
        CASE 
            WHEN bip.prediction_accuracy_score >= 80 THEN 5
            WHEN bip.prediction_accuracy_score >= 60 THEN 0
            ELSE -5
        END
    ))
    FROM public.blog_idea_performance bip
    WHERE bi.id = bip.blog_idea_id
    AND bip.prediction_accuracy_score IS NOT NULL;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.user_id()
 RETURNS uuid
 LANGUAGE sql
 SECURITY DEFINER
AS $function$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'sub',
    (current_setting('request.jwt.claims', true)::json->>'user_id')::text
  )::UUID;
$function$
;

create or replace view "public"."wordpress_ready_content" as  SELECT "Titles".id,
    "Titles".user_id,
    "Titles"."Title",
    "Titles"."userDescription",
    "Titles".seo_title_optimized,
    "Titles".seo_meta_desc_optimized,
    "Titles".wp_excerpt_auto_generated,
    "Titles".wp_slug,
    "Titles".focus_keyword,
    "Titles".primary_keywords_json,
    "Titles".secondary_keywords_json,
    "Titles".enhanced_primary_keywords,
    "Titles".total_search_volume,
    "Titles".traffic_potential_score,
    "Titles".content_format,
    "Titles".estimated_word_count,
    "Titles".content_outline,
    "Titles".key_points,
    "Titles".call_to_action_text,
    "Titles".wp_status,
    "Titles".priority_level,
    "Titles".scheduled_publish_date,
    "Titles".workflow_status,
    "Titles".overall_quality_score,
    "Titles".seo_optimization_score,
    "Titles"."dateCreatedOn",
    "Titles".last_updated
   FROM "Titles"
  WHERE (("Titles".workflow_status = ANY (ARRAY['ready_to_publish'::text, 'published'::text])) AND ("Titles".wp_status = ANY (ARRAY['publish'::text, 'pending'::text])));


CREATE OR REPLACE FUNCTION public.calculate_priority_score(search_volume integer, keyword_difficulty integer, cpc numeric, trend_percentage numeric, cpc_weight numeric DEFAULT 0.3, volume_weight numeric DEFAULT 0.4, trend_weight numeric DEFAULT 0.3)
 RETURNS numeric
 LANGUAGE plpgsql
AS $function$
DECLARE
    normalized_volume DECIMAL;
    normalized_difficulty DECIMAL;
    normalized_cpc DECIMAL;
    normalized_trend DECIMAL;
    priority_score DECIMAL;
BEGIN
    -- Normalize values to 0-100 scale
    normalized_volume := LEAST(100, GREATEST(0, (search_volume::DECIMAL / 10000) * 100));
    normalized_difficulty := 100 - keyword_difficulty; -- Invert difficulty (lower is better)
    normalized_cpc := LEAST(100, GREATEST(0, cpc * 10)); -- Scale CPC
    normalized_trend := LEAST(100, GREATEST(0, 50 + trend_percentage)); -- Center trend around 50
    
    -- Calculate weighted priority score
    priority_score := (
        normalized_volume * volume_weight +
        normalized_difficulty * (1 - cpc_weight - volume_weight - trend_weight) +
        normalized_cpc * cpc_weight +
        normalized_trend * trend_weight
    );
    
    RETURN ROUND(priority_score, 2);
END;
$function$
;

CREATE OR REPLACE FUNCTION public.cleanup_old_dataforseo_data()
 RETURNS integer
 LANGUAGE plpgsql
AS $function$
DECLARE
    deleted_count INTEGER := 0;
    temp_count INTEGER;
BEGIN
    -- Clean up old API logs (older than 30 days)
    DELETE FROM dataforseo_api_logs 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old trend data (older than 90 days)
    DELETE FROM trend_analysis_data 
    WHERE updated_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old keyword data (older than 30 days)
    DELETE FROM keyword_research_data 
    WHERE updated_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old suggestions (older than 7 days)
    DELETE FROM subtopic_suggestions 
    WHERE updated_at < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    RETURN deleted_count;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.get_dataforseo_stats()
 RETURNS jsonb
 LANGUAGE plpgsql
AS $function$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'trend_analysis_data', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'unique_subtopics', COUNT(DISTINCT subtopic),
                'unique_locations', COUNT(DISTINCT location),
                'avg_interest', ROUND(AVG(average_interest), 2),
                'max_interest', MAX(peak_interest),
                'last_updated', MAX(updated_at)
            )
            FROM trend_analysis_data
        ),
        'keyword_research_data', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'unique_keywords', COUNT(DISTINCT keyword),
                'avg_search_volume', ROUND(AVG(search_volume), 0),
                'avg_difficulty', ROUND(AVG(keyword_difficulty), 2),
                'avg_cpc', ROUND(AVG(cpc), 2),
                'last_updated', MAX(updated_at)
            )
            FROM keyword_research_data
        ),
        'subtopic_suggestions', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'trending_count', COUNT(*) FILTER (WHERE trending_status = 'TRENDING'),
                'stable_count', COUNT(*) FILTER (WHERE trending_status = 'STABLE'),
                'declining_count', COUNT(*) FILTER (WHERE trending_status = 'DECLINING'),
                'avg_growth_potential', ROUND(AVG(growth_potential), 2),
                'last_updated', MAX(updated_at)
            )
            FROM subtopic_suggestions
        ),
        'api_logs', (
            SELECT jsonb_build_object(
                'total_requests', COUNT(*),
                'successful_requests', COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 299),
                'failed_requests', COUNT(*) FILTER (WHERE status_code >= 400),
                'avg_response_time', ROUND(AVG(response_time_ms), 2),
                'last_request', MAX(created_at)
            )
            FROM dataforseo_api_logs
        )
    ) INTO stats;
    
    RETURN stats;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
 RETURNS trigger
 LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.validate_related_queries(data jsonb)
 RETURNS boolean
 LANGUAGE plpgsql
AS $function$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element is a string
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        IF jsonb_typeof(data->i) != 'string' THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    
    RETURN TRUE;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.validate_search_volume_trend(data jsonb)
 RETURNS boolean
 LANGUAGE plpgsql
AS $function$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element has required fields
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        DECLARE
            item JSONB := data->i;
        BEGIN
            IF NOT (item ? 'month' AND item ? 'volume') THEN
                RETURN FALSE;
            END IF;
            
            -- Check volume is numeric
            IF jsonb_typeof(item->'volume') != 'number' THEN
                RETURN FALSE;
            END IF;
        END;
    END LOOP;
    
    RETURN TRUE;
END;
$function$
;

CREATE OR REPLACE FUNCTION public.validate_timeline_data(data jsonb)
 RETURNS boolean
 LANGUAGE plpgsql
AS $function$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element has required fields
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        DECLARE
            item JSONB := data->i;
        BEGIN
            IF NOT (item ? 'date' AND item ? 'value') THEN
                RETURN FALSE;
            END IF;
            
            -- Check value is numeric
            IF jsonb_typeof(item->'value') != 'number' THEN
                RETURN FALSE;
            END IF;
        END;
    END LOOP;
    
    RETURN TRUE;
END;
$function$
;

create policy "Enable UPDATE for users based on user_id"
on "public"."PlannedArticles"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."PlannedArticles"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."PlannedArticles"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable selectfor users based on user_id"
on "public"."PlannedArticles"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Select  for users based on user_id"
on "public"."PostLinks"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."PostLinks"
as permissive
for update
to public
using ((auth.uid() = user_id))
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."PostLinks"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."PostLinks"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."RSS"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."RSS"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."RSS"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable users to view their own data only"
on "public"."RSS"
as permissive
for select
to authenticated
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."TableOfContents"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."TableOfContents"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable select for users based on user_id"
on "public"."TableOfContents"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "UPDATE for users based on user_id"
on "public"."TableOfContents"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."Titles"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."Titles"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."Titles"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable read access for auth users"
on "public"."Titles"
as permissive
for delete
to authenticated
using (true);


create policy "Enable select for users based on user_id"
on "public"."Titles"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Users can delete own titles"
on "public"."Titles"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own titles"
on "public"."Titles"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own titles"
on "public"."Titles"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own titles"
on "public"."Titles"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable delete for users based on user_id"
on "public"."Titles_citations"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for authenticated users only"
on "public"."Titles_citations"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable select  for users based on user_id"
on "public"."Titles_citations"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable update for users based on user_id"
on "public"."Titles_citations"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id))
with check (true);


create policy "Enable insert for authenticated users only"
on "public"."Tones"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable read access for authenticated users"
on "public"."Tones"
as permissive
for select
to authenticated
using (true);


create policy "Users can delete own affiliate offers"
on "public"."affiliate_offers"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own affiliate offers"
on "public"."affiliate_offers"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own affiliate offers"
on "public"."affiliate_offers"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own affiliate offers"
on "public"."affiliate_offers"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Admins can manage affiliate programs"
on "public"."affiliate_programs"
as permissive
for all
to public
using (is_admin());


create policy "Users can view affiliate programs"
on "public"."affiliate_programs"
as permissive
for select
to public
using (true);


create policy "Admins can view all affiliate research"
on "public"."affiliate_research"
as permissive
for select
to public
using (is_admin());


create policy "Users can delete own affiliate research"
on "public"."affiliate_research"
as permissive
for delete
to public
using ((user_id = user_id()));


create policy "Users can insert own affiliate research"
on "public"."affiliate_research"
as permissive
for insert
to public
with check ((user_id = user_id()));


create policy "Users can update own affiliate research"
on "public"."affiliate_research"
as permissive
for update
to public
using ((user_id = user_id()));


create policy "Users can view own affiliate research"
on "public"."affiliate_research"
as permissive
for select
to public
using ((user_id = user_id()));


create policy "Service role can manage all API keys"
on "public"."api_keys"
as permissive
for all
to public
using ((auth.role() = 'service_role'::text));


create policy "Enable Update for users based on user_id"
on "public"."application_settings"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."application_settings"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."application_settings"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable read access for all users"
on "public"."application_settings"
as permissive
for select
to authenticated
using (true);


create policy "Users can delete own generation results"
on "public"."blog_generation_results"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own generation results"
on "public"."blog_generation_results"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own generation results"
on "public"."blog_generation_results"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own generation results"
on "public"."blog_generation_results"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own keyword assignments"
on "public"."blog_idea_keyword_assignments"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert their own keyword assignments"
on "public"."blog_idea_keyword_assignments"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update their own keyword assignments"
on "public"."blog_idea_keyword_assignments"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own keyword assignments"
on "public"."blog_idea_keyword_assignments"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete own blog performance"
on "public"."blog_idea_performance"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own blog performance"
on "public"."blog_idea_performance"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own blog performance"
on "public"."blog_idea_performance"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own blog performance"
on "public"."blog_idea_performance"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete own templates"
on "public"."blog_idea_templates"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own templates"
on "public"."blog_idea_templates"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own templates"
on "public"."blog_idea_templates"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own and public templates"
on "public"."blog_idea_templates"
as permissive
for select
to public
using (((auth.uid() = user_id) OR (is_public = true) OR (user_id IS NULL)));


create policy "Users can delete own blog ideas"
on "public"."blog_ideas"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own blog ideas"
on "public"."blog_ideas"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own blog ideas"
on "public"."blog_ideas"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own blog ideas"
on "public"."blog_ideas"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable Select for users based on user_id"
on "public"."categoriesByPost"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."categoriesByPost"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."categoriesByPost"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."categoriesByPost"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Users can  update own competitive_intelligence"
on "public"."competitive_intelligence"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can delete own competitive_intelligence"
on "public"."competitive_intelligence"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert  own competitive_intelligence"
on "public"."competitive_intelligence"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can view own competitive_intelligence"
on "public"."competitive_intelligence"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete own content calendar"
on "public"."content_calendar"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own content calendar"
on "public"."content_calendar"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own content calendar"
on "public"."content_calendar"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own content calendar"
on "public"."content_calendar"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own content ideas"
on "public"."content_ideas"
as permissive
for delete
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Users can insert their own content ideas"
on "public"."content_ideas"
as permissive
for insert
to public
with check (((auth.uid())::text = (user_id)::text));


create policy "Users can update their own content ideas"
on "public"."content_ideas"
as permissive
for update
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Users can view their own content ideas"
on "public"."content_ideas"
as permissive
for select
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Enable all access for service role"
on "public"."content_opportunities"
as permissive
for all
to service_role
using (true)
with check (true);


create policy "Users can delete own content opportunities"
on "public"."content_opportunities"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own content opportunities"
on "public"."content_opportunities"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own content opportunities"
on "public"."content_opportunities"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own content opportunities"
on "public"."content_opportunities"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable delete for users based on user_id"
on "public"."embeddings"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."embeddings"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable read access for all users"
on "public"."embeddings"
as permissive
for select
to public
using (true);


create policy "Enable update access for all users"
on "public"."embeddings"
as permissive
for update
to public
using (true);


create policy "Users can delete own geographic_insights"
on "public"."geographic_insights"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own geographic_insights"
on "public"."geographic_insights"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own geographic_insights"
on "public"."geographic_insights"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own geographic_insights"
on "public"."geographic_insights"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own imported keywords"
on "public"."imported_keywords"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert their own imported keywords"
on "public"."imported_keywords"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update their own imported keywords"
on "public"."imported_keywords"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own imported keywords"
on "public"."imported_keywords"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Authenticated users can delete"
on "public"."indexed_documents"
as permissive
for delete
to authenticated
using (true);


create policy "Enable insert for authenticated users only"
on "public"."indexed_documents"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable read access for all users"
on "public"."indexed_documents"
as permissive
for select
to public
using (true);


create policy "Enable read access for all users"
on "public"."infographic"
as permissive
for select
to public
using (true);


create policy "Enable read access for all users"
on "public"."infographicDetails"
as permissive
for select
to public
using (true);


create policy "Enable all access for service role"
on "public"."keyword_intelligence"
as permissive
for all
to service_role
using (true)
with check (true);


create policy "Users can delete own keyword intelligence"
on "public"."keyword_intelligence"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own keyword intelligence"
on "public"."keyword_intelligence"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own keyword intelligence"
on "public"."keyword_intelligence"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own keyword intelligence"
on "public"."keyword_intelligence"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own keyword reports"
on "public"."keyword_opportunities_reports"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert their own keyword reports"
on "public"."keyword_opportunities_reports"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update their own keyword reports"
on "public"."keyword_opportunities_reports"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own keyword reports"
on "public"."keyword_opportunities_reports"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own keyword sessions"
on "public"."keyword_research_sessions"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert their own keyword sessions"
on "public"."keyword_research_sessions"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update their own keyword sessions"
on "public"."keyword_research_sessions"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own keyword sessions"
on "public"."keyword_research_sessions"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete their own keywords"
on "public"."keywords"
as permissive
for delete
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Users can insert their own keywords"
on "public"."keywords"
as permissive
for insert
to public
with check (((auth.uid())::text = (user_id)::text));


create policy "Users can update their own keywords"
on "public"."keywords"
as permissive
for update
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Users can view their own keywords"
on "public"."keywords"
as permissive
for select
to public
using (((auth.uid())::text = (user_id)::text));


create policy "Enable delete for users based on user_id"
on "public"."lindex_collections"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."lindex_collections"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable read access for all users"
on "public"."lindex_collections"
as permissive
for select
to public
using (true);


create policy "Enable update for users based on user_id"
on "public"."lindex_collections"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."lindex_documents"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."lindex_documents"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for authenticated users only"
on "public"."lindex_documents"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable read access for all users"
on "public"."lindex_documents"
as permissive
for select
to public
using (true);


create policy "Enable read access for all users"
on "public"."lindex_embedding_chunk"
as permissive
for select
to public
using (true);


create policy "Enable  Select for users based on user_id"
on "public"."lindex_sections"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable Update for users based on user_id"
on "public"."lindex_sections"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."lindex_sections"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."lindex_sections"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Admins can delete LLM configurations"
on "public"."llm_configurations"
as permissive
for delete
to public
using (is_admin());


create policy "Admins can insert LLM configurations"
on "public"."llm_configurations"
as permissive
for insert
to public
with check (is_admin());


create policy "Admins can update LLM configurations"
on "public"."llm_configurations"
as permissive
for update
to public
using (is_admin());


create policy "Admins can view LLM configurations"
on "public"."llm_configurations"
as permissive
for select
to public
using (is_admin());


create policy "Admins can delete LLM providers"
on "public"."llm_providers"
as permissive
for delete
to public
using (is_admin());


create policy "Admins can insert LLM providers"
on "public"."llm_providers"
as permissive
for insert
to public
with check (is_admin());


create policy "Admins can update LLM providers"
on "public"."llm_providers"
as permissive
for update
to public
using (is_admin());


create policy "Admins can view LLM providers"
on "public"."llm_providers"
as permissive
for select
to public
using (is_admin());


create policy "Users can delete own “manual_action_suggestions"
on "public"."manual_action_suggestions"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert  own “manual_action_suggestions"
on "public"."manual_action_suggestions"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update  own “manual_action_suggestions"
on "public"."manual_action_suggestions"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own “manual_action_suggestions"
on "public"."manual_action_suggestions"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable UPDATE for users based on user_id"
on "public"."mySources"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable delete for users based on user_id"
on "public"."mySources"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."mySources"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable users to view their own data only"
on "public"."mySources"
as permissive
for select
to authenticated
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Users can manage own offer analytics"
on "public"."offer_analytics"
as permissive
for all
to public
using ((user_id = user_id()));


create policy "Users can view own offer analytics"
on "public"."offer_analytics"
as permissive
for select
to public
using ((user_id = user_id()));


create policy "Users can manage own research sessions"
on "public"."offer_research_sessions"
as permissive
for all
to public
using ((user_id = user_id()));


create policy "Users can view own research sessions"
on "public"."offer_research_sessions"
as permissive
for select
to public
using ((user_id = user_id()));


create policy "Enable read access for all users"
on "public"."postTypes"
as permissive
for select
to public
using (true);


create policy "Users can delete their own research topics"
on "public"."research_topics"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert their own research topics"
on "public"."research_topics"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update their own research topics"
on "public"."research_topics"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view their own research topics"
on "public"."research_topics"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete own seasonal_calendar"
on "public"."seasonal_calendar"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert  own seasonal_calendar"
on "public"."seasonal_calendar"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can iupdate  own seasonal_calendar"
on "public"."seasonal_calendar"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own seasonal_calendar"
on "public"."seasonal_calendar"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable read access for all users"
on "public"."sectionSpecificPrompts"
as permissive
for select
to public
using (true);


create policy "Users can delete own topic decompositions"
on "public"."topic_decompositions"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own topic decompositions"
on "public"."topic_decompositions"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own topic decompositions"
on "public"."topic_decompositions"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own topic decompositions"
on "public"."topic_decompositions"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can delete own trend analyses"
on "public"."trend_analyses"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own trend analyses"
on "public"."trend_analyses"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own trend analyses"
on "public"."trend_analyses"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own trend analyses"
on "public"."trend_analyses"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Admins can view all trend analysis"
on "public"."trend_analysis"
as permissive
for select
to public
using (is_admin());


create policy "Users can delete own trend analysis"
on "public"."trend_analysis"
as permissive
for delete
to public
using ((user_id = user_id()));


create policy "Users can insert own trend analysis"
on "public"."trend_analysis"
as permissive
for insert
to public
with check ((user_id = user_id()));


create policy "Users can update own trend analysis"
on "public"."trend_analysis"
as permissive
for update
to public
using ((user_id = user_id()));


create policy "Users can view own trend analysis"
on "public"."trend_analysis"
as permissive
for select
to public
using ((user_id = user_id()));


create policy "Users can delete own trend_predictions"
on "public"."trend_predictions"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert  own trend_predictions"
on "public"."trend_predictions"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own trend_predictions"
on "public"."trend_predictions"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own trend_predictions"
on "public"."trend_predictions"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Enable all access for service role"
on "public"."trending_topics"
as permissive
for all
to service_role
using (true)
with check (true);


create policy "Users can delete own trending topics"
on "public"."trending_topics"
as permissive
for delete
to public
using ((auth.uid() = user_id));


create policy "Users can insert own trending topics"
on "public"."trending_topics"
as permissive
for insert
to public
with check ((auth.uid() = user_id));


create policy "Users can update own trending topics"
on "public"."trending_topics"
as permissive
for update
to public
using ((auth.uid() = user_id));


create policy "Users can view own trending topics"
on "public"."trending_topics"
as permissive
for select
to public
using ((auth.uid() = user_id));


create policy "Users can manage own offer preferences"
on "public"."user_offer_preferences"
as permissive
for all
to public
using ((user_id = user_id()));


create policy "Users can view own offer preferences"
on "public"."user_offer_preferences"
as permissive
for select
to public
using ((user_id = user_id()));


create policy "Insert Authenticated"
on "public"."user_profile"
as permissive
for insert
to authenticated
with check (((auth.jwt() ->> 'sub'::text) = (id)::text));


create policy "SelectPublic"
on "public"."user_profile"
as permissive
for select
to public
using (((auth.jwt() ->> 'sub'::text) = (id)::text));


create policy "Update Authenticated"
on "public"."user_profile"
as permissive
for update
to authenticated
using (((auth.jwt() ->> 'sub'::text) = (id)::text));


create policy "Admins can delete users"
on "public"."users"
as permissive
for delete
to public
using (is_admin());


create policy "Admins can insert users"
on "public"."users"
as permissive
for insert
to public
with check (is_admin());


create policy "Admins can update all users"
on "public"."users"
as permissive
for update
to public
using (is_admin());


create policy "Admins can view all users"
on "public"."users"
as permissive
for select
to public
using (is_admin());


create policy "Users can update own profile"
on "public"."users"
as permissive
for update
to public
using ((id = user_id()));


create policy "Users can view own profile"
on "public"."users"
as permissive
for select
to public
using ((id = user_id()));


create policy "Enable delete for users based on user_id"
on "public"."wordPress_details"
as permissive
for delete
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable insert for users based on user_id"
on "public"."wordPress_details"
as permissive
for insert
to public
with check ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable select for users based on user_id"
on "public"."wordPress_details"
as permissive
for select
to public
using ((( SELECT auth.uid() AS uid) = user_id));


create policy "Enable update for users based on email"
on "public"."wordPress_details"
as permissive
for update
to public
using ((( SELECT auth.uid() AS uid) = user_id))
with check ((( SELECT auth.uid() AS uid) = user_id));


CREATE TRIGGER update_titles_last_updated BEFORE UPDATE ON public."Titles" FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_affiliate_programs_updated_at BEFORE UPDATE ON public.affiliate_programs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_affiliate_research_updated_at BEFORE UPDATE ON public.affiliate_research FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_blog_performance_updated_at BEFORE UPDATE ON public.blog_idea_performance FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_blog_templates_updated_at BEFORE UPDATE ON public.blog_idea_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_blog_ideas_updated_at BEFORE UPDATE ON public.blog_ideas FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_calendar_updated_at BEFORE UPDATE ON public.content_calendar FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER embeddings AFTER INSERT ON public.embeddings FOR EACH ROW EXECUTE FUNCTION supabase_functions.http_request('https://dgcsqiaciyqvprtpopxg.supabase.co/functions/v1/generate-embeddings', 'POST', '{"Content-type":"application/json"}', '{}', '5000');

CREATE TRIGGER update_keyword_opportunity_score BEFORE INSERT OR UPDATE ON public.imported_keywords FOR EACH ROW EXECUTE FUNCTION trigger_update_opportunity_score();

CREATE TRIGGER update_session_keyword_stats AFTER INSERT OR DELETE ON public.imported_keywords FOR EACH ROW EXECUTE FUNCTION trigger_update_session_stats();

CREATE TRIGGER update_llm_configurations_updated_at BEFORE UPDATE ON public.llm_configurations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_llm_providers_updated_at BEFORE UPDATE ON public.llm_providers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_offer_analytics_updated_at BEFORE UPDATE ON public.offer_analytics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_offer_research_sessions_updated_at BEFORE UPDATE ON public.offer_research_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trend_analysis_updated_at BEFORE UPDATE ON public.trend_analysis FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_offer_preferences_updated_at BEFORE UPDATE ON public.user_offer_preferences FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


create schema if not exists "vecs";

create table "vecs"."RAG_HOUSE_AND_REAL_ESTATE_128D_OPT" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."default" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."general_knowledge" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."house" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."house_and_real_estate" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."rag_house" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."rag_house_and_real_estate" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."rag_house_and_real_estate_128D" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."rag_house_and_real_estate_128D_OPT" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."rag_house_and_real_estate_128d_opt" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."real_estate" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."real_estate_128D_OPT" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."research_documents" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."test" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."test-collection" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


create table "vecs"."your_collection" (
    "id" character varying not null,
    "vec" extensions.vector(128) not null,
    "metadata" jsonb not null default '{}'::jsonb
);


CREATE UNIQUE INDEX "RAG_HOUSE_AND_REAL_ESTATE_128D_OPT_pkey" ON vecs."RAG_HOUSE_AND_REAL_ESTATE_128D_OPT" USING btree (id);

CREATE UNIQUE INDEX default_pkey ON vecs."default" USING btree (id);

CREATE UNIQUE INDEX general_knowledge_pkey ON vecs.general_knowledge USING btree (id);

CREATE UNIQUE INDEX house_and_real_estate_pkey ON vecs.house_and_real_estate USING btree (id);

CREATE UNIQUE INDEX house_pkey ON vecs.house USING btree (id);

CREATE UNIQUE INDEX "rag_house_and_real_estate_128D_OPT_pkey" ON vecs."rag_house_and_real_estate_128D_OPT" USING btree (id);

CREATE UNIQUE INDEX "rag_house_and_real_estate_128D_pkey" ON vecs."rag_house_and_real_estate_128D" USING btree (id);

CREATE UNIQUE INDEX rag_house_and_real_estate_128d_opt_pkey ON vecs.rag_house_and_real_estate_128d_opt USING btree (id);

CREATE UNIQUE INDEX rag_house_and_real_estate_pkey ON vecs.rag_house_and_real_estate USING btree (id);

CREATE UNIQUE INDEX rag_house_pkey ON vecs.rag_house USING btree (id);

CREATE UNIQUE INDEX "real_estate_128D_OPT_pkey" ON vecs."real_estate_128D_OPT" USING btree (id);

CREATE UNIQUE INDEX real_estate_pkey ON vecs.real_estate USING btree (id);

CREATE UNIQUE INDEX research_documents_pkey ON vecs.research_documents USING btree (id);

CREATE UNIQUE INDEX "test-collection_pkey" ON vecs."test-collection" USING btree (id);

CREATE UNIQUE INDEX test_pkey ON vecs.test USING btree (id);

CREATE UNIQUE INDEX your_collection_pkey ON vecs.your_collection USING btree (id);

alter table "vecs"."RAG_HOUSE_AND_REAL_ESTATE_128D_OPT" add constraint "RAG_HOUSE_AND_REAL_ESTATE_128D_OPT_pkey" PRIMARY KEY using index "RAG_HOUSE_AND_REAL_ESTATE_128D_OPT_pkey";

alter table "vecs"."default" add constraint "default_pkey" PRIMARY KEY using index "default_pkey";

alter table "vecs"."general_knowledge" add constraint "general_knowledge_pkey" PRIMARY KEY using index "general_knowledge_pkey";

alter table "vecs"."house" add constraint "house_pkey" PRIMARY KEY using index "house_pkey";

alter table "vecs"."house_and_real_estate" add constraint "house_and_real_estate_pkey" PRIMARY KEY using index "house_and_real_estate_pkey";

alter table "vecs"."rag_house" add constraint "rag_house_pkey" PRIMARY KEY using index "rag_house_pkey";

alter table "vecs"."rag_house_and_real_estate" add constraint "rag_house_and_real_estate_pkey" PRIMARY KEY using index "rag_house_and_real_estate_pkey";

alter table "vecs"."rag_house_and_real_estate_128D" add constraint "rag_house_and_real_estate_128D_pkey" PRIMARY KEY using index "rag_house_and_real_estate_128D_pkey";

alter table "vecs"."rag_house_and_real_estate_128D_OPT" add constraint "rag_house_and_real_estate_128D_OPT_pkey" PRIMARY KEY using index "rag_house_and_real_estate_128D_OPT_pkey";

alter table "vecs"."rag_house_and_real_estate_128d_opt" add constraint "rag_house_and_real_estate_128d_opt_pkey" PRIMARY KEY using index "rag_house_and_real_estate_128d_opt_pkey";

alter table "vecs"."real_estate" add constraint "real_estate_pkey" PRIMARY KEY using index "real_estate_pkey";

alter table "vecs"."real_estate_128D_OPT" add constraint "real_estate_128D_OPT_pkey" PRIMARY KEY using index "real_estate_128D_OPT_pkey";

alter table "vecs"."research_documents" add constraint "research_documents_pkey" PRIMARY KEY using index "research_documents_pkey";

alter table "vecs"."test" add constraint "test_pkey" PRIMARY KEY using index "test_pkey";

alter table "vecs"."test-collection" add constraint "test-collection_pkey" PRIMARY KEY using index "test-collection_pkey";

alter table "vecs"."your_collection" add constraint "your_collection_pkey" PRIMARY KEY using index "your_collection_pkey";


CREATE TRIGGER newusertrigger AFTER INSERT ON auth.users FOR EACH ROW EXECUTE FUNCTION handle_new_user();


  create policy "Allow uploads to LlamaIndex folder"
  on "storage"."objects"
  as permissive
  for insert
  to authenticated
with check (((bucket_id = 'LlamaIndex'::text) AND ((storage.foldername(name))[1] = 'LlamaIndex'::text)));



  create policy "Article Images yg0r8i_0"
  on "storage"."objects"
  as permissive
  for select
  to public
using (((bucket_id = 'User Files'::text) AND ('articleImages'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Article Images yg0r8i_1"
  on "storage"."objects"
  as permissive
  for insert
  to public
with check (((bucket_id = 'User Files'::text) AND ('articleImages'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Article Images yg0r8i_2"
  on "storage"."objects"
  as permissive
  for update
  to public
using (((bucket_id = 'User Files'::text) AND ('articleImages'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Give Insert access to a folder 181v2yl_0"
  on "storage"."objects"
  as permissive
  for insert
  to public
with check (((bucket_id = 'LlamaIndex'::text) AND ((storage.foldername(name))[1] = 'LlamaIndex'::text) AND (( SELECT (auth.uid())::text AS uid) = 'd7bed83c-44a0-4a4f-925f-efc384ea1e50'::text)));



  create policy "Give access to a folder 181v2yl_0"
  on "storage"."objects"
  as permissive
  for select
  to public
using (((bucket_id = 'LlamaIndex'::text) AND ((storage.foldername(name))[1] = 'LlamaIndex'::text) AND (( SELECT (auth.uid())::text AS uid) = 'd7bed83c-44a0-4a4f-925f-efc384ea1e50'::text)));



  create policy "Give access to a folder x3c1vc_0"
  on "storage"."objects"
  as permissive
  for insert
  to public
with check (((bucket_id = 'App Files'::text) AND ((storage.foldername(name))[1] = 'admin'::text) AND ((storage.foldername(name))[2] = 'Infographics'::text) AND (( SELECT (auth.uid())::text AS uid) = 'd7bed83c-44a0-4a4f-925f-efc384ea1e50'::text)));



  create policy "Give access to a folder x3c1vc_1"
  on "storage"."objects"
  as permissive
  for update
  to public
using (((bucket_id = 'App Files'::text) AND ((storage.foldername(name))[1] = 'admin'::text) AND ((storage.foldername(name))[2] = 'Infographics'::text) AND (( SELECT (auth.uid())::text AS uid) = 'd7bed83c-44a0-4a4f-925f-efc384ea1e50'::text)));



  create policy "Give access to a folder x3c1vc_2"
  on "storage"."objects"
  as permissive
  for delete
  to public
using (((bucket_id = 'App Files'::text) AND ((storage.foldername(name))[1] = 'admin'::text) AND ((storage.foldername(name))[2] = 'Infographics'::text) AND (( SELECT (auth.uid())::text AS uid) = 'd7bed83c-44a0-4a4f-925f-efc384ea1e50'::text)));



  create policy "Give users authenticated access to folder x3c1vc_0"
  on "storage"."objects"
  as permissive
  for select
  to public
using (((bucket_id = 'App Files'::text) AND ((storage.foldername(name))[1] = 'private'::text) AND (auth.role() = 'authenticated'::text)));



  create policy "Llama Index yg0r8i_0"
  on "storage"."objects"
  as permissive
  for select
  to authenticated
using (((bucket_id = 'User Files'::text) AND ('llamaIndex'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Llama Index yg0r8i_1"
  on "storage"."objects"
  as permissive
  for insert
  to authenticated
with check (((bucket_id = 'User Files'::text) AND ('llamaIndex'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Llama Index yg0r8i_2"
  on "storage"."objects"
  as permissive
  for update
  to authenticated
using (((bucket_id = 'User Files'::text) AND ('llamaIndex'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



  create policy "Llama Index yg0r8i_3"
  on "storage"."objects"
  as permissive
  for delete
  to authenticated
using (((bucket_id = 'User Files'::text) AND ('llamaIndex'::text = (storage.foldername(name))[1]) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[2])));



