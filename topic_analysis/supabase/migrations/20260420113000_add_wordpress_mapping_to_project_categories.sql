alter table public.project_categories
    add column if not exists wordpress_category_id integer,
    add column if not exists wordpress_parent_category_id integer,
    add column if not exists wordpress_site_domain text,
    add column if not exists wordpress_last_synced_at timestamp with time zone;

create index if not exists idx_project_categories_wordpress_category_id
    on public.project_categories(wordpress_category_id);
