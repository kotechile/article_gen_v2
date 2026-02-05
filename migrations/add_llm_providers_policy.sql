-- Enable public read access for llm_providers
CREATE POLICY "Allow public read access for llm_providers" 
ON public.llm_providers
FOR SELECT 
USING (true);

-- Ensure RLS is enabled
ALTER TABLE public.llm_providers ENABLE ROW LEVEL SECURITY;
