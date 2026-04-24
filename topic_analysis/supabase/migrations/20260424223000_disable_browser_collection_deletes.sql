drop policy if exists "Enable delete for users based on user_id" on public.lindex_collections;

create policy "Disallow direct browser deletes on lindex_collections"
on public.lindex_collections
as restrictive
for delete
to public
using (false);
