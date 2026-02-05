import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://sbcontent.giniloh.com';
const supabaseKey = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJzdXBhYmFzZSIsImlhdCI6MTc2NDYxMjY2MCwiZXhwIjo0OTIwMjg2MjYwLCJyb2xlIjoiYW5vbiJ9.4z_OjFo4hYnh1RpOVGWJYWGWW1dWfSUtKs5w06H9PYI';

const supabase = createClient(supabaseUrl, supabaseKey);

async function checkSeoColumn() {
    console.log('Checking seo_optimization_score column...');
    // Try to select it specifically
    const { data, error } = await supabase
        .from('Titles')
        .select('id, seo_optimization_score, Title')
        .limit(3);

    if (error) {
        console.log(`Column 'seo_optimization_score': DOES NOT EXIST (or error: ${error.message})`);
    } else {
        console.log(`Column 'seo_optimization_score': EXISTS`);
        console.log('Sample data:', JSON.stringify(data, null, 2));
    }
}

checkSeoColumn();
