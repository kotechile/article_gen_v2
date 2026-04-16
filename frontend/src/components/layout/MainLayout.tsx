import { Sidebar } from "./Sidebar"
import { Outlet } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"

export default function MainLayout() {
    return (
        <div className="min-h-screen w-full bg-background text-foreground md:flex">
            <Sidebar />
            <main className="min-h-screen flex-1 overflow-auto md:pt-0">
                <Outlet />
            </main>
            <Toaster />
        </div>
    )
}
