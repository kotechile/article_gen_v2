"use client"

import * as React from "react"
import { Trash2, BookOpen, Code2, MoreHorizontal, FileText, Target, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog'
import { ContentIdea } from '@/types/idea-burst'

interface IdeaCardProps {
    idea: ContentIdea;
    selected: boolean;
    onSelect: (checked: boolean) => void;
    onDelete?: (id: string, e: React.MouseEvent) => void;
}

export function IdeaCard({ idea, selected, onSelect, onDelete }: IdeaCardProps) {
    if (!idea) return null;

    // Determine icon based on content type
    const Icon = idea.content_type === 'software' ? Code2 : BookOpen

    // Safety check for title
    const displayTitle = idea.title || 'Untitled Idea';
    const displayDescription = idea.description || 'No description available';
    const volume = idea.total_search_volume || 0;

    return (
        <Dialog>
            <Card className={cn("relative group transition-all hover:shadow-md cursor-pointer",
                selected ? 'ring-2 ring-primary border-primary' : ''
            )}>
                <div className="absolute top-3 right-3 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                    {onDelete && (
                        <Button
                            variant="destructive"
                            size="icon"
                            className="h-8 w-8 rounded-full shadow-sm"
                            onClick={(e) => {
                                e.stopPropagation();
                                onDelete(idea.id, e);
                            }}
                        >
                            <Trash2 className="h-4 w-4" />
                        </Button>
                    )}
                    <Checkbox
                        checked={selected}
                        onChange={(e) => onSelect(e.target.checked)}
                        className="h-5 w-5 rounded-full shadow-sm bg-background"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>

                <DialogTrigger asChild>
                    <div className="h-full">
                        <CardHeader className="pb-3">
                            <div className="flex items-start justify-between gap-4">
                                <div className="space-y-1">
                                    <Badge variant="outline" className="w-fit mb-2 flex items-center gap-1.5">
                                        <Icon className="h-3 w-3" />
                                        {idea.content_type === 'software' ? 'Micro-SaaS' : 'Blog Post'}
                                    </Badge>
                                    <CardTitle className="text-lg leading-tight group-hover:text-primary transition-colors">
                                        {displayTitle}
                                    </CardTitle>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm text-muted-foreground line-clamp-3 mb-4">
                                {displayDescription}
                            </p>

                            <div className="flex items-center gap-4 text-xs text-muted-foreground mt-auto">
                                <div className="flex items-center gap-1">
                                    <Target className="h-3.5 w-3.5" />
                                    <span>Vol: {volume.toLocaleString()}</span>
                                </div>
                                {idea.monetization_hook && (
                                    <div className="flex items-center gap-1 text-green-600 dark:text-green-400">
                                        <Zap className="h-3.5 w-3.5" />
                                        <span>Monetized</span>
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </div>
                </DialogTrigger>
            </Card>

            <DialogContent className="max-w-2xl">
                <DialogHeader>
                    <DialogTitle className="text-2xl flex items-center gap-2">
                        <Icon className="h-6 w-6 text-primary" />
                        {displayTitle}
                    </DialogTitle>
                    <DialogDescription>
                        Generated on {new Date(idea.created_at || Date.now()).toLocaleDateString()}
                    </DialogDescription>
                </DialogHeader>

                <div className="grid gap-6 py-4">
                    <div className="space-y-4">
                        <div>
                            <h4 className="text-sm font-medium text-muted-foreground mb-1">Description</h4>
                            <p className="text-base">{displayDescription}</p>
                        </div>

                        {idea.content_outline && idea.content_outline.length > 0 && (
                            <div>
                                <h4 className="text-sm font-medium text-muted-foreground mb-2 flex items-center gap-2">
                                    <FileText className="h-4 w-4" />
                                    Suggested Outline
                                </h4>
                                <ul className="space-y-2">
                                    {idea.content_outline.map((point: string, i: number) => (
                                        <li key={i} className="flex items-start gap-2 text-sm bg-muted/50 p-2 rounded-md">
                                            <span className="font-medium text-primary/70">{i + 1}.</span>
                                            {point}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                            <div>
                                <h4 className="text-sm font-medium text-muted-foreground mb-1">Target Keywords</h4>
                                <div className="flex flex-wrap gap-1">
                                    {/* Combine all keywords for display */}
                                    {[...(idea.primary_keywords || []), ...(idea.secondary_keywords || []), ...(idea.keywords || [])]
                                        .filter((v: any, i, a) => {
                                            // Custom uniqueness filter that handles objects if necessary or roughly by string rep
                                            // For simplicity, just checking index uniqueness of the raw item
                                            return a.indexOf(v) === i;
                                        })
                                        .slice(0, 10) // Limit count
                                        .map((k: any, i: number) => {
                                            const keywordText = typeof k === 'string' ? k : (k.keyword || k.seed_keyword || JSON.stringify(k));
                                            return (
                                                <Badge key={i} variant="secondary" className="text-xs">
                                                    {keywordText}
                                                </Badge>
                                            );
                                        })}
                                    {/* Fallback if absolutely no keywords */}
                                    {(!idea.primary_keywords?.length && !idea.secondary_keywords?.length && !idea.keywords?.length) && (
                                        <span className="text-xs text-muted-foreground italic">No specific keywords</span>
                                    )}
                                </div>
                            </div>

                            {idea.monetization_hook && (
                                <div>
                                    <h4 className="text-sm font-medium text-muted-foreground mb-1">Monetization</h4>
                                    <p className="text-sm text-green-600 dark:text-green-400 font-medium">
                                        {idea.monetization_hook}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    )
}
