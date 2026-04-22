"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Lightbulb } from "lucide-react";
import { Subtopic } from "@/types/research";

interface SubtopicsGridProps {
    subtopics: Subtopic[];
    selectedSubtopics: Set<string>;
    onToggle: (id: string) => void;
}

export function SubtopicsGrid({ subtopics, selectedSubtopics, onToggle }: SubtopicsGridProps) {
    if (!subtopics || subtopics.length === 0) {
        return (
            <div className="text-center py-12 text-muted-foreground">
                No subtopics found. Click &quot;Decompose Topic&quot; to generate ideas.
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {subtopics.map((subtopic) => {
                const rationale =
                    subtopic.rationale ||
                    subtopic.trend_analysis?.rationale ||
                    "Editorial subtopic generated for expansion into content ideas.";
                const keywords = (subtopic.keywords || subtopic.seed_keywords || []).slice(0, 3);
                const offerCount =
                    subtopic.monetization_data?.offers?.length ??
                    subtopic.monetization_data?.offer_count ??
                    subtopic.affiliate_offer_count ??
                    0;
                const offerLabel = offerCount === 1 ? "1 Offer" : `${offerCount} Offers`;

                return (
                    <Card
                        key={subtopic.id}
                        className={`relative border-zinc-200 dark:border-zinc-800 bg-background/50 ${
                            selectedSubtopics.has(subtopic.id) ? "ring-2 ring-primary border-primary" : ""
                        }`}
                    >
                        <div className="absolute right-3 top-3">
                            <Checkbox
                                checked={selectedSubtopics.has(subtopic.id)}
                                onChange={() => onToggle(subtopic.id)}
                            />
                        </div>
                        <CardHeader className="pr-12">
                            <CardTitle className="flex items-start gap-2 text-lg leading-tight">
                                <Lightbulb className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                                <span>{subtopic.name}</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-3">
                            <p className="line-clamp-3 text-sm text-muted-foreground">{rationale}</p>
                            {keywords.length > 0 && (
                                <div className="flex flex-wrap gap-1.5">
                                    {keywords.map((keyword: unknown, idx: number) => (
                                        <Badge key={`${subtopic.id}-kw-${idx}`} variant="secondary" className="text-xs">
                                            {typeof keyword === "string"
                                                ? keyword
                                                : (keyword as { keyword?: string }).keyword || "keyword"}
                                        </Badge>
                                    ))}
                                </div>
                            )}
                            <div className="text-xs text-muted-foreground">
                                {offerLabel}
                            </div>
                        </CardContent>
                    </Card>
                );
            })}
        </div>
    );
}
