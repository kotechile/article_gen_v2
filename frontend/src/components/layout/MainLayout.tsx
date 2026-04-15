import { Sidebar } from "./Sidebar"
import { Outlet } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"
import { ThemeProvider } from "@/components/theme-provider"

export default function MainLayout() {
    return (
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
            <div className="min-h-screen w-full bg-[#08101d] text-white md:flex">
                <Sidebar />
                <main className="min-h-screen flex-1 overflow-auto pt-20 md:pt-0">
                    <Outlet />
                </main>
                <Toaster />
            </div>
        </ThemeProvider>
    )
}
