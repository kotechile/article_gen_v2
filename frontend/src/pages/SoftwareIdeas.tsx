import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Trash2, Wrench, Star } from 'lucide-react'
import { motion } from 'framer-motion'
import { useAuth } from '@/context/auth-context'
import { supabase } from '@/lib/supabase'
import { Button } from '@/components/ui/button'
import { contentIdeasService } from '@/services/content-ideas.service'

type SoftwareIdea = {
    id: string
    title: string
    description: string | null
    status: string | null
    published: boolean | null
    created_at: string
    topic_id: string | null
    topic_rating?: number | null
}

export function SoftwareIdeas() {
    const { user } = useAuth()
    const navigate = useNavigate()
    const [ideas, setIdeas] = React.useState<SoftwareIdea[]>([])
    const [search, setSearch] = React.useState('')
    const [loading, setLoading] = React.useState(true)

    const loadIdeas = React.useCallback(async () => {
        if (!user) return
        try {
            setLoading(true)
            const { data, error } = await supabase
                .from('content_ideas')
                .select('*')
                .eq('user_id', user.id)
                .eq('content_type', 'software')
                .order('created_at', { ascending: false })

            if (error) throw error
            console.info('[SoftwareIdeas] data loaded', { count: (data || []).length })
            const normalized = ((data || []) as any[]).map((row) => ({
                id: row.id,
                title: row.title,
                description: row.description ?? null,
                status: row.status ?? null,
                published: row.published ?? null,
                created_at: row.created_at,
                topic_id: row.topic_id ?? null,
                topic_rating: row.topic_rating ?? 0,
            })) as SoftwareIdea[]
            setIdeas(normalized)
        } catch (err) {
            console.error('Failed to load software ideas:', err)
        } finally {
            setLoading(false)
        }
    }, [user])

    React.useEffect(() => {
        loadIdeas()
    }, [loadIdeas])

    const filtered = React.useMemo(
        () => ideas.filter((idea) => idea.title.toLowerCase().includes(search.toLowerCase())),
        [ideas, search]
    )

    const handleDelete = async (id: string) => {
        if (!confirm('Delete this software idea?')) return
        const { error } = await supabase.from('content_ideas').delete().eq('id', id)
        if (error) {
            console.error('Failed to delete software idea:', error)
            return
        }
        setIdeas((prev) => prev.filter((idea) => idea.id !== id))
    }

    const handleSetRating = async (id: string, rating: number) => {
        if (!user) return
        const nextRating = Math.max(0, Math.min(5, Number(rating || 0)))
        const ok = await contentIdeasService.updateContentIdeaRating(id, user.id, nextRating)
        if (!ok) return
        setIdeas((prev) => prev.map((idea) => (idea.id === id ? { ...idea, topic_rating: nextRating } : idea)))
    }

    return (
        <div className="min-h-screen bg-background">
            <div className="mx-auto max-w-6xl px-8 py-10 lg:py-14">
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                    className="mb-8"
                >
                    <h1 className="text-2xl font-semibold tracking-tight text-foreground">Software Ideas</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Saved software concepts for future build and validation.
                    </p>
                </motion.div>

                <div className="relative mb-6 max-w-sm">
                    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search software ideas..."
                        className="h-10 w-full rounded-lg border border-border bg-muted/50 pl-10 pr-4 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring/50"
                    />
                </div>

                {loading ? (
                    <div className="space-y-3">
                        {[1, 2, 3].map((idx) => (
                            <div key={idx} className="h-28 animate-pulse rounded-xl bg-muted" />
                        ))}
                    </div>
                ) : filtered.length === 0 ? (
                    <div className="rounded-2xl border border-border bg-muted/20 p-12 text-center text-muted-foreground">
                        No saved software ideas yet.
                    </div>
                ) : (
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                        {filtered.map((idea) => {
                            const status = idea.published || idea.status?.toLowerCase() === 'published' ? 'Published' : 'Saved'
                            return (
                                <div key={idea.id} className="rounded-xl border border-border bg-muted/30 p-5">
                                    <div className="mb-3 flex items-start justify-between gap-3">
                                        <div className="flex items-start gap-2">
                                            <Wrench className="mt-0.5 h-4 w-4 text-primary" />
                                            <h3 className="text-base font-semibold text-foreground">{idea.title}</h3>
                                        </div>
                                        <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2 py-0.5 text-xs text-cyan-300">
                                            {status}
                                        </span>
                                    </div>

                                    {idea.description && (
                                        <p className="mb-4 line-clamp-3 text-sm text-muted-foreground">{idea.description}</p>
                                    )}

                                    <div className="mb-4 flex items-center gap-1">
                                        {[1, 2, 3, 4, 5].map((value) => {
                                            const isFilled = value <= Number(idea.topic_rating || 0)
                                            return (
                                                <button
                                                    key={`${idea.id}-rating-${value}`}
                                                    type="button"
                                                    onClick={() => handleSetRating(idea.id, value)}
                                                    className="rounded p-0.5 transition hover:scale-105"
                                                    aria-label={`Rate ${idea.title} ${value} star${value === 1 ? '' : 's'}`}
                                                >
                                                    <Star className={`h-4 w-4 ${isFilled ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground'}`} />
                                                </button>
                                            )
                                        })}
                                    </div>

                                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                                        <span>{new Date(idea.created_at).toLocaleDateString()}</span>
                                        <div className="flex items-center gap-2">
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="h-8 border-border"
                                                onClick={() => navigate(`/content-studio?id=${idea.id}`)}
                                            >
                                                Open
                                            </Button>
                                            <button
                                                type="button"
                                                onClick={() => handleDelete(idea.id)}
                                                className="rounded-md p-1.5 text-muted-foreground transition hover:bg-muted hover:text-destructive"
                                                title="Delete"
                                            >
                                                <Trash2 className="h-3.5 w-3.5" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                )}
            </div>
        </div>
    )
}
