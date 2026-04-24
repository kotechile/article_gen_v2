alter table if exists projects
    add column if not exists brand_primary_color text,
    add column if not exists brand_text_color text,
    add column if not exists brand_secondary_color text,
    add column if not exists brand_neutral_color text,
    add column if not exists branding_updated_at timestamptz;

alter table if exists "wordPress_details"
    add column if not exists brand_primary_color text,
    add column if not exists brand_text_color text,
    add column if not exists brand_secondary_color text,
    add column if not exists brand_neutral_color text,
    add column if not exists branding_updated_at timestamptz;
