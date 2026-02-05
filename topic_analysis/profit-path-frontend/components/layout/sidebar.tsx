"use client"

import * as React from "react"
import { LayoutDashboard, Settings, History, TrendingUp, ChevronLeft, ChevronRight, LogOut, User } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ModeToggle } from "@/components/mode-toggle"
import { Separator } from "@/components/ui/separator"
import { useAuth } from "@/context/auth-context"
import Link from "next/link"
import { usePathname } from "next/navigation"

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> { }

export function Sidebar({ className }: SidebarProps) {
    const [isCollapsed, setIsCollapsed] = React.useState(false)
    const pathname = usePathname()

    return (
        <div className={cn("relative h-screen border-r bg-background transition-all duration-300", isCollapsed ? "w-[80px]" : "w-[240px]", className)}>
            <div className="flex h-full flex-col">
                {/* Header / Logo */}
                <div className={cn("flex items-center h-16 px-4", isCollapsed ? "justify-center" : "justify-between")}>
                    {!isCollapsed && (
                        <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-zinc-900 to-zinc-500 bg-clip-text text-transparent dark:from-white dark:to-zinc-500">
                            ProfitPath
                        </span>
                    )}
                    {isCollapsed && (
                        <span className="text-xl font-bold">PP</span>
                    )}
                </div>

                <Separator />

                {/* Navigation */}
                <ScrollArea className="flex-1 px-3 py-4">
                    <div className="space-y-2">
                        <NavItem
                            href="/"
                            icon={<LayoutDashboard className="h-5 w-5" />}
                            label="Dashboard"
                            isCollapsed={isCollapsed}
                            active={pathname === "/"}
                        />
                        <NavItem
                            href="/research"
                            icon={<TrendingUp className="h-5 w-5" />}
                            label="Research"
                            isCollapsed={isCollapsed}
                            active={pathname?.startsWith("/research")}
                        />
                        <NavItem
                            href="/history"
                            icon={<History className="h-5 w-5" />}
                            label="History"
                            isCollapsed={isCollapsed}
                            active={pathname?.startsWith("/history")}
                        />
                        <NavItem
                            href="/settings/research"
                            icon={<Settings className="h-5 w-5" />}
                            label="Settings"
                            isCollapsed={isCollapsed}
                            active={pathname?.startsWith("/settings")}
                        />
                    </div>
                </ScrollArea>

                {/* Footer */}
                <div className="p-3 mt-auto border-t space-y-2">
                    <UserProfile isCollapsed={isCollapsed} />

                    <Separator />

                    <div className={cn("flex items-center", isCollapsed ? "justify-center" : "justify-between px-2")}>
                        {!isCollapsed && <span className="text-sm text-muted-foreground">Theme</span>}
                        <ModeToggle />
                    </div>

                    <div className="flex justify-center pt-2">
                        <Button variant="ghost" size="sm" className="w-full" onClick={() => setIsCollapsed(!isCollapsed)}>
                            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <div className="flex items-center text-muted-foreground text-xs"><ChevronLeft className="h-3 w-3 mr-1" /> Collapse</div>}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}

function NavItem({ icon, label, isCollapsed, active, href }: { icon: React.ReactNode, label: string, isCollapsed: boolean, active?: boolean, href: string }) {
    return (
        <Button
            asChild
            variant={active ? "secondary" : "ghost"}
            className={cn("w-full justify-start", isCollapsed ? "justify-center px-0" : "px-4")}
            title={isCollapsed ? label : undefined}
        >
            <Link href={href}>
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
                <Button variant="ghost" size="icon" className="rounded-full h-8 w-8 bg-primary/10 hover:bg-primary/20" title={user.email || 'User'}>
                    <span className="text-xs font-bold text-primary">{initials}</span>
                </Button>
                <Button variant="ghost" size="icon" onClick={signOut} title="Sign Out">
                    <LogOut className="h-4 w-4" />
                </Button>
            </div>
        )
    }

    return (
        <div className="space-y-2 px-2">
            <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center h-9 w-9 rounded-full bg-primary/10 border border-primary/20 shadow-sm">
                    <span className="text-sm font-bold text-primary">{initials}</span>
                </div>
                <div className="flex-1 overflow-hidden">
                    <p className="text-sm font-medium truncate">{user.email}</p>
                </div>
            </div>
            <Button variant="outline" size="sm" className="w-full justify-start px-3 h-9 mt-1" onClick={signOut}>
                <LogOut className="h-3.5 w-3.5 mr-2" />
                Sign Out
            </Button>
        </div>
    )
}
