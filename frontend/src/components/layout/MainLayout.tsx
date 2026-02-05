import { Sidebar } from "./Sidebar"
import { Outlet } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"
import { ThemeProvider } from "@/components/theme-provider"

export default function MainLayout() {
    return (
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
            <div className="flex h-screen w-full bg-background text-foreground">
                <Sidebar />
                <main className="flex-1 overflow-auto">
                    <Outlet />
                </main>
                <Toaster />
            </div>
        </ThemeProvider>
    )
}
