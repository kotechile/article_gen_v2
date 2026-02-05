-- Check if topic_id and user_id columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'keyword_research_data' 
AND column_name IN ('topic_id', 'user_id');
