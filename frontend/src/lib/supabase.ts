
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Missing Supabase environment variables');
  // Display error on screen
  const errorDiv = document.createElement('div');
  errorDiv.style.cssText = 'position:fixed;top:0;left:0;z-index:9999;background:red;color:white;padding:20px;width:100%;font-weight:bold;';
  errorDiv.textContent = 'CRITICAL: Missing Supabase environment variables (VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY). Check frontend/.env';
  document.body.appendChild(errorDiv);
}

// Use placeholders to prevent top-level crash, allowing the app to at least render the error
export const supabase = createClient(supabaseUrl || 'https://placeholder.supabase.co', supabaseAnonKey || 'placeholder');
