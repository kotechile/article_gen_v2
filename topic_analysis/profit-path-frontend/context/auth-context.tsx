"use client"

import * as React from "react"
import { Session, User } from "@supabase/supabase-js"
import { supabase } from "@/lib/supabase"
import { useRouter, usePathname } from "next/navigation"

interface AuthContextType {
    session: Session | null
    user: User | null
    isLoading: boolean
    signOut: () => Promise<void>
}

const AuthContext = React.createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [session, setSession] = React.useState<Session | null>(null)
    const [user, setUser] = React.useState<User | null>(null)
    const [isLoading, setIsLoading] = React.useState(true)
    const router = useRouter()
    const pathname = usePathname()

    React.useEffect(() => {
        if (!isLoading && !session) {
            // List of public paths that don't require authentication
            const isPublicPath = pathname === '/login' || pathname?.startsWith('/auth');

            if (!isPublicPath) {
                console.log("AuthProvider: No session found on protected route, redirecting to /login");
                router.push('/login');
            }
        }
    }, [isLoading, session, pathname, router])

    const pathnameRef = React.useRef(pathname)

    React.useEffect(() => {
        pathnameRef.current = pathname
    }, [pathname])

    React.useEffect(() => {
        // 1. Get initial session
        const initializeAuth = async () => {
            try {
                const { data: { session }, error } = await supabase.auth.getSession()
                if (error) throw error

                console.log("AuthProvider: Initial session retrieved", session ? "User found" : "No user");
                setSession(session)
                setUser(session?.user ?? null)
            } catch (error) {
                console.error("Auth initialization error:", error)
            } finally {
                setIsLoading(false)
            }
        }

        initializeAuth()

        // 2. Listen for auth changes
        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((_event, session) => {
            console.log(`AuthProvider: Auth state changed: ${_event}`, session ? "Session active" : "No session");
            setSession(session)
            setUser(session?.user ?? null)
            setIsLoading(false)

            if (_event === 'SIGNED_IN') {
                // Only redirect to dashboard if we are currently on an auth page
                // This prevents resetting the user to home if they are already on a protected page (e.g. /research/[id])
                // and a session refresh triggers SIGNED_IN
                const currentPath = pathnameRef.current;
                if (currentPath === '/login' || currentPath?.startsWith('/auth')) {
                    console.log("AuthProvider: Redirecting to / from auth page");
                    router.push('/')
                }
            }

            if (_event === 'SIGNED_OUT') {
                console.log("AuthProvider: Redirecting to /login");
                router.push('/login')
            }
        })

        return () => subscription.unsubscribe()
    }, [router])

    const signOut = async () => {
        await supabase.auth.signOut()
        router.push('/login')
    }

    const value = {
        session,
        user,
        isLoading,
        signOut,
    }

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
    const context = React.useContext(AuthContext)
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider")
    }
    return context
}
