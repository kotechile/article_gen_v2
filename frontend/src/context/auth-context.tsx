"use client"

import * as React from "react"
import { type Session, type User } from "@supabase/supabase-js"
import { supabase } from "@/lib/supabase"
import { useNavigate, useLocation } from "react-router-dom"

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
    const navigate = useNavigate()
    const location = useLocation()
    const pathname = location.pathname

    React.useEffect(() => {
        if (!isLoading && !session) {
            // List of public paths that don't require authentication
            const isPublicPath = pathname === '/login' || pathname?.startsWith('/auth');

            if (!isPublicPath) {
                console.log("AuthProvider: No session found on protected route, redirecting to /login");
                navigate('/login');
            }
        }
    }, [isLoading, session, pathname, navigate])

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
        } = supabase.auth.onAuthStateChange((_event: string, session: Session | null) => {
            console.log(`AuthProvider: Auth state changed: ${_event}`, session ? "Session active" : "No session");
            setSession(session)
            setUser(session?.user ?? null)
            setIsLoading(false)

            if (_event === 'SIGNED_IN') {
                // Only redirect to dashboard if we are currently on an auth page
                const currentPath = pathnameRef.current;
                if (currentPath === '/login' || currentPath?.startsWith('/auth')) {
                    console.log("AuthProvider: Redirecting to / from auth page");
                    navigate('/')
                }
            }

            if (_event === 'SIGNED_OUT') {
                console.log("AuthProvider: Redirecting to /login");
                navigate('/login')
            }
        })

        return () => subscription.unsubscribe()
    }, [navigate])

    const signOut = async () => {
        await supabase.auth.signOut()
        navigate('/login')
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
