
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

console.log('Supabase Config Check:');
console.log('URL:', supabaseUrl);
console.log('Key (first 10 chars):', supabaseAnonKey ? supabaseAnonKey.substring(0, 10) : 'MISSING');

if (!supabaseUrl || !supabaseAnonKey) {
  console.error('Missing Supabase environment variables');
  // Display error on screen
  const errorDiv = document.createElement('div');
  errorDiv.style.cssText = 'position:fixed;top:0;left:0;z-index:9999;background:red;color:white;padding:20px;width:100%;font-weight:bold;';
  errorDiv.textContent = 'CRITICAL: Missing Supabase environment variables (VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY). Check frontend/.env';
  document.body.appendChild(errorDiv);
}

// Use placeholders to prevent top-level crash, allowing the app to at least render the error
export const supabase = createClient(supabaseUrl || 'https://placeholder.supabase.co', supabaseAnonKey || 'placeholder', {
  auth: {
    flowType: 'implicit',
    persistSession: true,
    detectSessionInUrl: true,
    autoRefreshToken: true,
    debug: true
  }
});

export function isAuthError(err: any): boolean {
  if (!err) return false;
  const status = err.status || err.statusCode || (typeof err.code === 'number' ? err.code : null);
  const code = String(err.code || '');
  const message = String(err.message || '');
  const details = String(err.details || '');
  const hint = String(err.hint || '');

  if (status === 401 || code === '401' || code === 'PGRST301') {
    return true;
  }

  const text = `${code} ${message} ${details} ${hint}`.toLowerCase();
  return (
    text.includes('jwt') ||
    text.includes('token') ||
    text.includes('unauthorized') ||
    text.includes('expired') ||
    text.includes('invalid claim') ||
    text.includes('session not found')
  );
}

let refreshPromise: Promise<any> | null = null;

export async function refreshSupabaseSession() {
  if (refreshPromise) {
    return refreshPromise;
  }
  refreshPromise = (async () => {
    try {
      const { data, error } = await supabase.auth.refreshSession();
      if (error) {
        const { data: sData } = await supabase.auth.getSession();
        return sData?.session ?? null;
      }
      return data?.session ?? null;
    } catch (e) {
      console.warn('[SupabaseAuth] Error refreshing session:', e);
      try {
        const { data: sData } = await supabase.auth.getSession();
        return sData?.session ?? null;
      } catch {
        return null;
      }
    } finally {
      refreshPromise = null;
    }
  })();
  return refreshPromise;
}

export async function getFreshSession() {
  try {
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error || !session) {
      return await refreshSupabaseSession();
    }
    // If expires within 60 seconds
    if (session.expires_at && session.expires_at * 1000 < Date.now() + 60000) {
      return await refreshSupabaseSession();
    }
    return session;
  } catch (err) {
    console.warn('[SupabaseAuth] getFreshSession fallback to refresh:', err);
    return await refreshSupabaseSession();
  }
}

/**
 * Executes a Supabase operation with automatic session refresh and retries on 401 / auth errors.
 */
export async function withSessionRetry<T>(
  operation: () => Promise<T>,
  maxRetries = 3,
  delayMs = 500
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      const result = await operation();
      if (result && typeof result === 'object') {
        if (Array.isArray(result)) {
          const authErrorResult = result.find(
            (r) => r && typeof r === 'object' && 'error' in r && isAuthError(r.error)
          );
          if (authErrorResult && attempt < maxRetries) {
            attempt++;
            console.warn(`[SupabaseAuth] Auth error in batch response (attempt ${attempt}/${maxRetries}), refreshing session...`, authErrorResult.error);
            await refreshSupabaseSession();
            await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
            continue;
          }
        } else if ('error' in result && isAuthError((result as any).error)) {
          if (attempt < maxRetries) {
            attempt++;
            console.warn(`[SupabaseAuth] Auth error in response (attempt ${attempt}/${maxRetries}), refreshing session...`, (result as any).error);
            await refreshSupabaseSession();
            await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
            continue;
          }
        }
      }
      return result;
    } catch (err: any) {
      if (isAuthError(err) && attempt < maxRetries) {
        attempt++;
        console.warn(`[SupabaseAuth] Auth exception caught (attempt ${attempt}/${maxRetries}), refreshing session...`, err);
        await refreshSupabaseSession();
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
        continue;
      }
      throw err;
    }
  }
}

