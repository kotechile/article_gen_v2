-- Enable public read access for llm_providers_image
CREATE POLICY "Allow public read access for llm_providers_image" 
ON public.llm_providers_image
FOR SELECT 
USING (true);

-- Also ensure RLS is enabled (which it already is based on your screenshot)
ALTER TABLE public.llm_providers_image ENABLE ROW LEVEL SECURITY;
