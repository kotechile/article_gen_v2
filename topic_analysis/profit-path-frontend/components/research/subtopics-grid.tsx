"use client";

import { IdeaCard } from "@/components/burst/idea-card";
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
                No subtopics found. Click "Decompose Topic" to generate ideas.
            </div>
        );
    }

    return (
        <div className="columns-1 md:columns-2 lg:columns-3 gap-4">
            {subtopics.map((subtopic) => {
                // Same robust logic as SubtopicsTable to ensure count consistency
                const offerCount = subtopic.monetization_data?.offers?.length
                    ?? subtopic.monetization_data?.offer_count
                    ?? subtopic.affiliate_offer_count
                    ?? 0;

                // Map subtopic to ContentIdea shape for usage in IdeaCard
                const ideaProps: any = {
                    id: subtopic.id,
                    title: subtopic.name,
                    description: `${offerCount} affiliate offers available`,
                    content_type: 'article', // Default type for subtopics
                    total_search_volume: subtopic.search_volume ?? 0,
                    average_difficulty: subtopic.seo_difficulty ?? 0,
                    viability_score: subtopic.viability_score ?? 0,
                    monetization_hook: offerCount > 0 ? `${offerCount} Offers` : undefined,
                    created_at: new Date().toISOString(),
                    keywords: subtopic.keywords || subtopic.seed_keywords || [], // Pass keywords to detail modal (backend uses seed_keywords)
                    // trending info is not standard in ContentIdea but adaptable if needed
                };

                return (
                    <IdeaCard
                        key={subtopic.id}
                        idea={ideaProps}
                        selected={selectedSubtopics.has(subtopic.id)}
                        onSelect={(checked) => onToggle(subtopic.id)}
                    // No delete action for subtopics in grid view for now
                    />
                );
            })}
        </div>
    );
}
