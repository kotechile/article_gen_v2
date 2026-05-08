create table if not exists public.released_software_ideas (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    source_idea_id uuid,
    topic_id text,
    title text not null,
    description text,
    status text not null default 'saved',
    released_at timestamp with time zone not null default now(),
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now(),
    topic_rating integer default 0,
    published boolean not null default true,
    content_type text not null default 'software',
    subtopic text,
    category text,
    domain text,
    keywords text[] default '{}'::text[],
    primary_keywords jsonb not null default '[]'::jsonb,
    secondary_keywords jsonb not null default '[]'::jsonb,
    search_phrase text,
    total_search_volume integer,
    average_difficulty numeric,
    average_cpc numeric,
    affiliate_offer_count integer,
    viability_score integer,
    trend_score integer,
    monetization_score integer,
    seo_ease_score integer,
    opportunity_score integer,
    product_type text,
    user_job_to_be_done text,
    key_inputs jsonb not null default '[]'::jsonb,
    output_result text,
    build_complexity text,
    distribution_angle text,
    target_intent text,
    content_outline text[] default '{}'::text[],
    ranking_breakdown jsonb not null default '{}'::jsonb,
    keyword_metrics jsonb not null default '{}'::jsonb,
    idea_metadata jsonb not null default '{}'::jsonb,
    raw_dataforseo_output jsonb,
    raw_supabase_output jsonb
);

create unique index if not exists released_software_ideas_user_source_idx
    on public.released_software_ideas (user_id, source_idea_id)
    where source_idea_id is not null;

create index if not exists released_software_ideas_user_released_at_idx
    on public.released_software_ideas (user_id, released_at desc);

create or replace function public.update_released_software_ideas_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists update_released_software_ideas_timestamp on public.released_software_ideas;

create trigger update_released_software_ideas_timestamp
before update on public.released_software_ideas
for each row
execute function public.update_released_software_ideas_updated_at();

alter table public.released_software_ideas enable row level security;

drop policy if exists "released_software_ideas_select_own" on public.released_software_ideas;
create policy "released_software_ideas_select_own"
on public.released_software_ideas
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "released_software_ideas_insert_own" on public.released_software_ideas;
create policy "released_software_ideas_insert_own"
on public.released_software_ideas
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "released_software_ideas_update_own" on public.released_software_ideas;
create policy "released_software_ideas_update_own"
on public.released_software_ideas
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "released_software_ideas_delete_own" on public.released_software_ideas;
create policy "released_software_ideas_delete_own"
on public.released_software_ideas
for delete
to authenticated
using (auth.uid() = user_id);
