"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Send, X } from "lucide-react"

interface FloatingActionBarProps {
    selectedCount: number
    onClear: () => void
    onPublish: () => void
}

export function FloatingActionBar({ selectedCount, onClear, onPublish }: FloatingActionBarProps) {
    return (
        <AnimatePresence>
            {selectedCount > 0 && (
                <motion.div
                    initial={{ y: 100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: 100, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 pointer-events-none"
                >
                    <div className="bg-zinc-900/90 dark:bg-zinc-100/90 backdrop-blur-md text-zinc-50 dark:text-zinc-900 px-6 py-3 rounded-full shadow-2xl flex items-center gap-6 pointer-events-auto border border-zinc-800 dark:border-zinc-200">
                        <div className="flex items-center gap-2">
                            <span className="font-bold text-lg">{selectedCount}</span>
                            <span className="text-sm font-medium opacity-80">selected</span>
                        </div>

                        <div className="h-6 w-px bg-current opacity-20" />

                        <div className="flex items-center gap-2">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={onClear}
                                className="hover:bg-white/10 text-current hover:text-current h-8"
                            >
                                Clear
                            </Button>
                            <Button
                                size="sm"
                                onClick={onPublish}
                                className="rounded-full px-5 h-9 bg-white text-black hover:bg-zinc-200 dark:bg-zinc-900 dark:text-white dark:hover:bg-zinc-800"
                            >
                                Publish <Send className="ml-2 w-3.5 h-3.5" />
                            </Button>
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    )
}
