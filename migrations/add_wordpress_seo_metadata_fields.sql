alter table if exists projects
    add column if not exists seo_plugin text,
    add column if not exists site_url_override text,
    add column if not exists social_default_image_url text;

alter table if exists "wordPress_details"
    add column if not exists seo_plugin text,
    add column if not exists site_url_override text,
    add column if not exists social_default_image_url text;
