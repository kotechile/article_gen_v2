import * as React from "react"
import { Settings, TrendingUp, ChevronLeft, ChevronRight, LogOut, Home, BookOpen, Menu } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ModeToggle } from "@/components/mode-toggle"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/context/auth-context"
import { Link, useLocation } from "react-router-dom"
import { supabase } from "@/lib/supabase"
import { ZenithLogo } from "./ZenithLogo"

const SIDEBAR_STORAGE_KEY = "zenith_sidebar_collapsed"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> { }

export function Sidebar({ className }: SidebarProps) {
    const { user } = useAuth()
    const [isCollapsed, setIsCollapsed] = React.useState(false)
    const [mobileOpen, setMobileOpen] = React.useState(false)
    const [recentArticles, setRecentArticles] = React.useState<Array<{ id: string; Title: string }>>([])
    const location = useLocation()
    const pathname = location.pathname

    React.useEffect(() => {
        const saved = window.localStorage.getItem(SIDEBAR_STORAGE_KEY)
        if (saved) {
            setIsCollapsed(saved === "true")
        }
    }, [])

    React.useEffect(() => {
        window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(isCollapsed))
    }, [isCollapsed])

    React.useEffect(() => {
        setMobileOpen(false)
    }, [pathname])

    React.useEffect(() => {
        const loadRecentArticles = async () => {
            if (!user) {
                setRecentArticles([])
                return
            }

            const { data, error } = await supabase
                .from("Titles")
                .select("id, Title")
                .eq("user_id", user.id)
                .order("dateCreatedOn", { ascending: false })
                .limit(4)

            if (error) {
                console.error("Failed to load recent articles", error)
                return
            }

            setRecentArticles(data || [])
        }

        loadRecentArticles()
    }, [user])

    const toggleCollapsed = () => setIsCollapsed((current) => !current)

    return (
        <>
            <button
                type="button"
                onClick={() => setMobileOpen(true)}
                className="fixed left-4 top-4 z-50 inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-slate-950/80 text-white shadow-lg backdrop-blur md:hidden"
                aria-label="Open navigation"
            >
                <Menu className="h-5 w-5" />
            </button>

            {mobileOpen && (
                <button
                    type="button"
                    className="fixed inset-0 z-30 bg-slate-950/65 backdrop-blur-sm md:hidden"
                    onClick={() => setMobileOpen(false)}
                    aria-label="Close navigation overlay"
                />
            )}

            <aside
                className={cn(
                    "fixed inset-y-0 left-0 z-40 flex h-screen flex-col border-r border-white/8 bg-[#0b1324]/95 backdrop-blur-xl transition-all duration-300 md:static md:translate-x-0",
                    mobileOpen ? "translate-x-0" : "-translate-x-full",
                    isCollapsed ? "w-[92px]" : "w-[290px]",
                    className,
                )}
            >
                <div className="flex h-full flex-col">
                    <div className={cn("flex h-16 items-center px-5", isCollapsed ? "justify-center" : "justify-start")}>
                        {isCollapsed ? (
                            <ZenithLogo compact />
                        ) : (
                            <ZenithLogo />
                        )}
                    </div>

                    <Separator className="bg-white/6" />

                    <ScrollArea className="flex-1 px-3 py-5">
                        <div className="space-y-2">
                            <NavItem
                                href="/"
                                icon={<Home className="h-5 w-5" />}
                                label="New Research"
                                isCollapsed={isCollapsed}
                                active={pathname === "/"}
                            />
                            <NavItem
                                href="/research"
                                icon={<TrendingUp className="h-5 w-5" />}
                                label="All Research"
                                isCollapsed={isCollapsed}
                                active={pathname?.startsWith("/research")}
                            />
                            <NavItem
                                href="/my-articles"
                                icon={<BookOpen className="h-5 w-5" />}
                                label="Content Library"
                                isCollapsed={isCollapsed}
                                active={pathname?.startsWith("/my-articles")}
                            />
                            <NavItem
                                href="/settings"
                                icon={<Settings className="h-5 w-5" />}
                                label="Settings"
                                isCollapsed={isCollapsed}
                                active={pathname?.startsWith("/settings")}
                            />
                        </div>

                        {!isCollapsed && (
                            <div className="mt-8 rounded-2xl border border-white/8 bg-white/4 px-3 py-3">
                                <div className="mb-2 flex items-center justify-between">
                                    <p className="text-xs font-medium text-slate-400">Recent Articles</p>
                                    <Link to="/my-articles" className="text-[11px] text-slate-500 transition-colors hover:text-slate-300">
                                        View all
                                    </Link>
                                </div>

                                <div className="space-y-0.5">
                                    {recentArticles.length === 0 && (
                                        <p className="px-1 py-3 text-[11px] text-slate-600">
                                            No articles yet.
                                        </p>
                                    )}

                                    {recentArticles.map((article) => (
                                        <Link
                                            key={article.id}
                                            to="/my-articles"
                                            className="block truncate rounded-lg px-2 py-1.5 text-sm text-slate-400 transition-colors hover:bg-white/[0.04] hover:text-white"
                                        >
                                            {article.Title || "Untitled Article"}
                                        </Link>
                                    ))}
                                </div>
                            </div>
                        )}
                    </ScrollArea>

                    <div className="mt-auto space-y-3 border-t border-white/6 p-3">
                        <UserProfile isCollapsed={isCollapsed} />

                        <div className={cn("flex items-center", isCollapsed ? "justify-center" : "justify-between px-2")}>
                            {!isCollapsed && <span className="text-sm text-slate-500">Theme</span>}
                            <ModeToggle />
                        </div>

                        <Button
                            variant="ghost"
                            size="sm"
                            className={cn(
                                "w-full rounded-2xl border border-white/6 bg-white/4 text-slate-300 hover:bg-white/8 hover:text-white",
                                isCollapsed ? "px-0" : "justify-start px-4",
                            )}
                            onClick={toggleCollapsed}
                        >
                            {isCollapsed ? (
                                <ChevronRight className="h-4 w-4" />
                            ) : (
                                <>
                                    <ChevronLeft className="mr-2 h-4 w-4" />
                                    Collapse
                                </>
                            )}
                        </Button>
                    </div>
                </div>
            </aside>
        </>
    )
}

function NavItem({ icon, label, isCollapsed, active, href }: { icon: React.ReactNode, label: string, isCollapsed: boolean, active?: boolean, href: string }) {
    return (
        <Button
            asChild
            variant={active ? "secondary" : "ghost"}
            className={cn(
                "w-full rounded-2xl border text-slate-300 transition-all",
                active
                    ? "border-blue-400/20 bg-blue-500/15 text-white hover:bg-blue-500/20"
                    : "border-transparent bg-transparent hover:border-white/8 hover:bg-white/5 hover:text-white",
                isCollapsed ? "justify-center px-0" : "justify-start px-4",
            )}
            title={isCollapsed ? label : undefined}
        >
            <Link to={href}>
                {icon}
                {!isCollapsed && <span className="ml-3 font-medium">{label}</span>}
            </Link>
        </Button>
    )
}

function UserProfile({ isCollapsed }: { isCollapsed: boolean }) {
    const { user, signOut } = useAuth()

    if (!user) return null

    // Helper to get initials
    const getInitials = (user: any) => {
        const name = user.user_metadata?.full_name || user.user_metadata?.name || ""
        if (name) {
            return name
                .split(" ")
                .map((n: string) => n[0])
                .join("")
                .toUpperCase()
                .slice(0, 2)
        }
        return user.email?.slice(0, 2).toUpperCase() || "??"
    }

    const initials = getInitials(user)

    if (isCollapsed) {
        return (
            <div className="flex flex-col items-center space-y-2">
                <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full bg-blue-500/15 hover:bg-blue-500/20" title={user.email || 'User'}>
                    <span className="text-xs font-bold text-blue-200">{initials}</span>
                </Button>
                <Button variant="ghost" size="icon" onClick={signOut} title="Sign Out" className="text-slate-400 hover:text-white">
                    <LogOut className="h-4 w-4" />
                </Button>
            </div>
        )
    }

    return (
        <div className="space-y-2 px-2">
            <div className="flex items-center space-x-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full border border-blue-400/15 bg-blue-500/15 shadow-sm">
                    <span className="text-sm font-bold text-blue-200">{initials}</span>
                </div>
                <div className="flex-1 overflow-hidden">
                    <p className="truncate text-sm font-medium text-white">{user.email}</p>
                    <p className="text-xs text-slate-500">Signed in</p>
                </div>
            </div>
            <Button variant="outline" size="sm" className="mt-1 h-9 w-full justify-start border-white/10 bg-white/4 px-3 text-slate-300 hover:bg-white/8 hover:text-white" onClick={signOut}>
                <LogOut className="h-3.5 w-3.5 mr-2" />
                Sign Out
            </Button>
        </div>
    )
}
