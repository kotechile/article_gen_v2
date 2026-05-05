alter table if exists research_topics
  add column if not exists topic_mode text,
  add column if not exists keyword_viability_score numeric,
  add column if not exists keyword_viability_label text,
  add column if not exists topic_generation_reasoning text,
  add column if not exists topic_generation_metadata jsonb default '{}'::jsonb;

update research_topics
set
  topic_mode = coalesce(topic_mode, 'hybrid'),
  keyword_viability_label = coalesce(keyword_viability_label, 'medium'),
  topic_generation_metadata = coalesce(topic_generation_metadata, '{}'::jsonb)
where
  topic_mode is null
  or keyword_viability_label is null
  or topic_generation_metadata is null;

alter table if exists project_topic_candidates
  add column if not exists topic_mode text,
  add column if not exists keyword_viability_score numeric,
  add column if not exists keyword_viability_label text,
  add column if not exists topic_generation_reasoning text,
  add column if not exists topic_generation_metadata jsonb default '{}'::jsonb;

update project_topic_candidates
set
  topic_mode = coalesce(topic_mode, 'hybrid'),
  keyword_viability_label = coalesce(keyword_viability_label, 'medium'),
  topic_generation_metadata = coalesce(topic_generation_metadata, '{}'::jsonb)
where
  topic_mode is null
  or keyword_viability_label is null
  or topic_generation_metadata is null;

create index if not exists idx_research_topics_topic_mode on research_topics(topic_mode);
create index if not exists idx_research_topics_keyword_viability_label on research_topics(keyword_viability_label);
