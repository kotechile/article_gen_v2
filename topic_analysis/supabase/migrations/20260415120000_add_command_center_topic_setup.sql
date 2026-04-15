create table if not exists public.project_categories (
    id uuid primary key default gen_random_uuid(),
    project_id uuid not null references public.projects(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    name text not null,
    slug text not null,
    level integer not null check (level in (1, 2)),
    parent_category_id uuid references public.project_categories(id) on delete cascade,
    sort_order integer not null default 0,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now()
);

create index if not exists idx_project_categories_project_id on public.project_categories(project_id);
create index if not exists idx_project_categories_parent on public.project_categories(parent_category_id);
create unique index if not exists ux_project_categories_project_slug on public.project_categories(project_id, slug);

alter table public.project_categories enable row level security;

drop policy if exists "Users can view their own project categories" on public.project_categories;
create policy "Users can view their own project categories"
on public.project_categories
as permissive
for select
to public
using (auth.uid() = user_id);

drop policy if exists "Users can insert their own project categories" on public.project_categories;
create policy "Users can insert their own project categories"
on public.project_categories
as permissive
for insert
to public
with check (auth.uid() = user_id);

drop policy if exists "Users can update their own project categories" on public.project_categories;
create policy "Users can update their own project categories"
on public.project_categories
as permissive
for update
to public
using (auth.uid() = user_id);

drop policy if exists "Users can delete their own project categories" on public.project_categories;
create policy "Users can delete their own project categories"
on public.project_categories
as permissive
for delete
to public
using (auth.uid() = user_id);

create table if not exists public.project_topic_candidates (
    id uuid primary key default gen_random_uuid(),
    project_id uuid not null references public.projects(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    primary_category_id uuid not null references public.project_categories(id) on delete cascade,
    secondary_category_id uuid references public.project_categories(id) on delete cascade,
    title text not null,
    topic_source text not null default 'seed' check (topic_source in ('seed', 'ai', 'news', 'manual')),
    source_label text,
    created_at timestamp with time zone not null default now(),
    updated_at timestamp with time zone not null default now()
);

create index if not exists idx_project_topic_candidates_project_id on public.project_topic_candidates(project_id);
create index if not exists idx_project_topic_candidates_secondary_category_id on public.project_topic_candidates(secondary_category_id);
create unique index if not exists ux_project_topic_candidates_project_category_title
on public.project_topic_candidates (
    project_id,
    coalesce(secondary_category_id, '00000000-0000-0000-0000-000000000000'::uuid),
    lower(title)
);

alter table public.project_topic_candidates enable row level security;

drop policy if exists "Users can view their own project topic candidates" on public.project_topic_candidates;
create policy "Users can view their own project topic candidates"
on public.project_topic_candidates
as permissive
for select
to public
using (auth.uid() = user_id);

drop policy if exists "Users can insert their own project topic candidates" on public.project_topic_candidates;
create policy "Users can insert their own project topic candidates"
on public.project_topic_candidates
as permissive
for insert
to public
with check (auth.uid() = user_id);

drop policy if exists "Users can update their own project topic candidates" on public.project_topic_candidates;
create policy "Users can update their own project topic candidates"
on public.project_topic_candidates
as permissive
for update
to public
using (auth.uid() = user_id);

drop policy if exists "Users can delete their own project topic candidates" on public.project_topic_candidates;
create policy "Users can delete their own project topic candidates"
on public.project_topic_candidates
as permissive
for delete
to public
using (auth.uid() = user_id);

alter table public.research_topics
    add column if not exists project_id uuid references public.projects(id) on delete set null,
    add column if not exists primary_category_id uuid references public.project_categories(id) on delete set null,
    add column if not exists secondary_category_id uuid references public.project_categories(id) on delete set null,
    add column if not exists topic_source text,
    add column if not exists source_topic_id uuid references public.project_topic_candidates(id) on delete set null;

create index if not exists idx_research_topics_project_id on public.research_topics(project_id);
create index if not exists idx_research_topics_secondary_category_id on public.research_topics(secondary_category_id);

create or replace function public.seed_project_command_center_data(
    p_project_id uuid,
    p_user_id uuid,
    p_project_label text
)
returns void
language plpgsql
as $$
declare
    v_label text := coalesce(nullif(trim(p_project_label), ''), 'this website');
    v_parent_audience uuid;
    v_parent_commercial uuid;
    v_child_foundations uuid;
    v_child_seasonal uuid;
    v_child_trending uuid;
    v_child_comparisons uuid;
    v_child_buying uuid;
    v_child_optimization uuid;
begin
    insert into public.project_categories (project_id, user_id, name, slug, level, parent_category_id, sort_order)
    values (p_project_id, p_user_id, 'Audience Growth', 'audience-growth', 1, null, 10)
    on conflict do nothing;

    insert into public.project_categories (project_id, user_id, name, slug, level, parent_category_id, sort_order)
    values (p_project_id, p_user_id, 'Commercial Intent', 'commercial-intent', 1, null, 20)
    on conflict do nothing;

    select id into v_parent_audience
    from public.project_categories
    where project_id = p_project_id and level = 1 and slug = 'audience-growth'
    order by created_at
    limit 1;

    select id into v_parent_commercial
    from public.project_categories
    where project_id = p_project_id and level = 1 and slug = 'commercial-intent'
    order by created_at
    limit 1;

    insert into public.project_categories (project_id, user_id, name, slug, level, parent_category_id, sort_order)
    values
        (p_project_id, p_user_id, 'Foundational Guides', 'foundational-guides', 2, v_parent_audience, 10),
        (p_project_id, p_user_id, 'Seasonal Opportunities', 'seasonal-opportunities', 2, v_parent_audience, 20),
        (p_project_id, p_user_id, 'Trending Conversations', 'trending-conversations', 2, v_parent_audience, 30),
        (p_project_id, p_user_id, 'Product Comparisons', 'product-comparisons', 2, v_parent_commercial, 10),
        (p_project_id, p_user_id, 'Buying Advice', 'buying-advice', 2, v_parent_commercial, 20),
        (p_project_id, p_user_id, 'Maintenance & Optimization', 'maintenance-optimization', 2, v_parent_commercial, 30)
    on conflict do nothing;

    select id into v_child_foundations
    from public.project_categories
    where project_id = p_project_id and slug = 'foundational-guides'
    order by created_at
    limit 1;

    select id into v_child_seasonal
    from public.project_categories
    where project_id = p_project_id and slug = 'seasonal-opportunities'
    order by created_at
    limit 1;

    select id into v_child_trending
    from public.project_categories
    where project_id = p_project_id and slug = 'trending-conversations'
    order by created_at
    limit 1;

    select id into v_child_comparisons
    from public.project_categories
    where project_id = p_project_id and slug = 'product-comparisons'
    order by created_at
    limit 1;

    select id into v_child_buying
    from public.project_categories
    where project_id = p_project_id and slug = 'buying-advice'
    order by created_at
    limit 1;

    select id into v_child_optimization
    from public.project_categories
    where project_id = p_project_id and slug = 'maintenance-optimization'
    order by created_at
    limit 1;

    insert into public.project_topic_candidates (
        project_id,
        user_id,
        primary_category_id,
        secondary_category_id,
        title,
        topic_source,
        source_label
    )
    values
        (p_project_id, p_user_id, v_parent_audience, v_child_foundations, 'Getting started with ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_audience, v_child_foundations, 'Common mistakes people make with ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_audience, v_child_seasonal, 'Seasonal planning ideas for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_audience, v_child_seasonal, 'What to prioritize this quarter for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_audience, v_child_trending, 'Emerging trends shaping ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_audience, v_child_trending, 'What people are asking right now about ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_comparisons, 'Best tools and products for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_comparisons, 'Top options to compare for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_buying, 'How to choose the right solution for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_buying, 'What to look for before buying for ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_optimization, 'How to maintain and improve ' || v_label, 'seed', 'Starter Topic'),
        (p_project_id, p_user_id, v_parent_commercial, v_child_optimization, 'Performance upgrades and optimization ideas for ' || v_label, 'seed', 'Starter Topic')
    on conflict do nothing;
end;
$$;

create or replace function public.handle_project_command_center_seed()
returns trigger
language plpgsql
as $$
begin
    perform public.seed_project_command_center_data(
        new.id,
        new.user_id,
        coalesce(new.domain, new.app_name, new.site_description, new.websiteDescription)
    );
    return new;
end;
$$;

drop trigger if exists trg_seed_project_command_center_data on public.projects;

create trigger trg_seed_project_command_center_data
after insert on public.projects
for each row
execute function public.handle_project_command_center_seed();

do $$
declare
    project_record record;
begin
    for project_record in
        select
            id,
            user_id,
            coalesce(domain, app_name, site_description, websiteDescription) as project_label
        from public.projects
    loop
        perform public.seed_project_command_center_data(
            project_record.id,
            project_record.user_id,
            project_record.project_label
        );
    end loop;
end;
$$;
