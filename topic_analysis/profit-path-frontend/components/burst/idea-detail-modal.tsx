"use client";

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from "react-markdown";

interface IdeaDetailModalProps {
    idea: {
        id: string;
        title: string;
        description?: string;
        content_type: string;
        markdown_outline?: string;
        target_audience?: string;
        content_angle?: string;
        keywords?: string[];
    } | null;
    isOpen: boolean;
    onClose: () => void;
}

export function IdeaDetailModal({ idea, isOpen, onClose }: IdeaDetailModalProps) {
    if (!idea) return null;

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-4xl max-h-[80vh]">
                <DialogHeader>
                    <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                            <Badge variant="secondary" className="mb-2">
                                {idea.content_type.toUpperCase()}
                            </Badge>
                            <DialogTitle className="text-2xl">{idea.title}</DialogTitle>
                            {idea.description && (
                                <DialogDescription className="mt-2 text-base">
                                    {idea.description}
                                </DialogDescription>
                            )}
                        </div>
                    </div>
                </DialogHeader>

                <ScrollArea className="max-h-[60vh]">
                    <div className="space-y-6 pr-4">
                        {/* Metadata */}
                        {(idea.target_audience || idea.content_angle) && (
                            <div className="grid grid-cols-2 gap-4 p-4 bg-muted/50 rounded-lg">
                                {idea.target_audience && (
                                    <div>
                                        <div className="text-sm font-medium text-muted-foreground">
                                            Target Audience
                                        </div>
                                        <div className="text-sm mt-1">{idea.target_audience}</div>
                                    </div>
                                )}
                                {idea.content_angle && (
                                    <div>
                                        <div className="text-sm font-medium text-muted-foreground">
                                            Content Angle
                                        </div>
                                        <div className="text-sm mt-1">{idea.content_angle}</div>
                                    </div>
                                )}
                            </div>
                        )}


                        {/* Content Outline */}
                        <div>
                            <h3 className="text-lg font-semibold mb-3">Content Outline</h3>
                            {idea.markdown_outline && idea.markdown_outline !== "## Outline Not Generated" ? (
                                <div className="prose prose-sm max-w-none dark:prose-invert">
                                    <ReactMarkdown>{idea.markdown_outline}</ReactMarkdown>
                                </div>
                            ) : (
                                <div className="text-sm text-muted-foreground italic p-4 bg-muted/30 rounded-lg">
                                    No structured outline available for this idea. The outline feature
                                    may not have been enabled when this idea was generated.
                                </div>
                            )}
                        </div>

                        {/* Target Keywords */}
                        <div>
                            <h3 className="text-lg font-semibold mb-3">Target Keywords</h3>
                            {idea.keywords && idea.keywords.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {idea.keywords.map((keyword: any, index) => {
                                        const keywordText = typeof keyword === 'string' ? keyword : (keyword.keyword || keyword.seed_keyword || JSON.stringify(keyword));
                                        return (
                                            <Badge key={index} variant="outline" className="text-sm">
                                                {keywordText}
                                            </Badge>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="text-sm text-muted-foreground italic">
                                    No specific keywords targeted for this idea.
                                </div>
                            )}
                        </div>
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
