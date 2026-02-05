import { NextRequest, NextResponse } from 'next/server'
import { createServerClient, type CookieOptions } from '@supabase/ssr'

export async function GET(request: NextRequest) {
    const requestUrl = new URL(request.url)
    const code = requestUrl.searchParams.get('code')

    console.log('Auth Callback: Received request to', requestUrl.toString())

    if (code) {
        console.log('Auth Callback: Code found, exchanging for session...')
        // Use Host header to correctly handle Traefik/Nginx proxying
        // requestUrl.origin might be internal (0.0.0.0:3000)
        // Use Host header to correctly handle Traefik/Nginx proxying
        // requestUrl.origin might be internal (0.0.0.0:3000)
        let host = request.headers.get('x-forwarded-host') || request.headers.get('host');

        // Safety Fallback: If host is internal or localhost, force the production domain
        if (!host || host.includes('0.0.0.0') || host.includes('localhost') || host.includes('backend')) {
            console.log("Auth Callback: Host header was invalid (" + host + "), forcing ideas.aichieve.net");
            host = 'ideas.aichieve.net';
        }

        const protocol = request.headers.get('x-forwarded-proto') || 'https';
        const baseUrl = `https://${host}`; // Force HTTPS for production

        console.log(`Auth Callback: Redirecting to calculated base: ${baseUrl}`);
        const response = NextResponse.redirect(new URL('/', baseUrl))
        const supabase = createServerClient(
            process.env.NEXT_PUBLIC_SUPABASE_URL!,
            process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            {
                cookies: {
                    get(name: string) {
                        return request.cookies.get(name)?.value
                    },
                    set(name: string, value: string, options: CookieOptions) {
                        response.cookies.set({
                            name,
                            value,
                            ...options,
                        })
                    },
                    remove(name: string, options: CookieOptions) {
                        response.cookies.set({
                            name,
                            value: '',
                            ...options,
                        })
                    },
                },
            }
        )

        const { error } = await supabase.auth.exchangeCodeForSession(code)

        if (error) {
            console.error('Auth Callback ERROR:', error.message)
            return NextResponse.redirect(new URL(`/login?error=${encodeURIComponent(error.message)}`, request.url))
        }

        console.log('Auth Callback: SUCCESS, redirecting to dashboard')
        return response
    }

    console.log('Auth Callback: No code found in URL')
    return NextResponse.redirect(new URL('/login', request.url))
}
