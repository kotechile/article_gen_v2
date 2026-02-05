import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://sbcontent.giniloh.com';
const supabaseKey = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzdXBhYmFzZSIsImlhdCI6MTc2NDYxMjY2MCwiZXhwIjo0OTIwMjg2MjYwLCJyb2xlIjoiYW5vbiJ9.4z_OjFo4hYnh1RpOVGWJYWGWW1dWfSUtKs5w06H9PYI';

const supabase = createClient(supabaseUrl, supabaseKey);

async function checkColumn(tableName, colName) {
    const { data, error } = await supabase
        .from(tableName)
        .select(colName)
        .limit(1);

    if (error) {
        console.log(`Column '${tableName}.${colName}': DOES NOT EXIST (or error: ${error.message})`);
    } else {
        console.log(`Column '${tableName}.${colName}': EXISTS`);
    }
}

async function runChecks() {
    console.log('Checking content_ideas columns...');
    await checkColumn('content_ideas', 'keyword_metrics');
    await checkColumn('content_ideas', 'markdown_outline');

    console.log('Checking Titles columns again...');
    await checkColumn('Titles', 'content_outline');
}

runChecks();
