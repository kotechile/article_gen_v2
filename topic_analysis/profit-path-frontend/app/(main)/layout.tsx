import { Sidebar } from "@/components/layout/sidebar";

export default function MainLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <div className="flex min-h-screen bg-muted/40 dark:bg-zinc-950">
            {/* Fixed Sidebar */}
            <aside className="hidden h-screen border-r bg-background lg:block">
                <Sidebar />
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto">
                <div className="flex h-16 items-center gap-4 border-b bg-background px-6 lg:hidden">
                    <div className="font-bold">ProfitPath</div>
                    {/* Mobile trigger would go here */}
                </div>
                <div className="flex-1 p-8 pt-6">
                    {children}
                </div>
            </main>
        </div>
    );
}
