export interface Collection {
    id: string;
    name: string;
    user_id: string;
    created_at?: string;
}

export interface Document {
    id: string | number;
    title: string;
    source_type: string;
    content?: string;
    parsedText?: string;
    url?: string;
    in_vector_store?: boolean;
    processing_status?: 'pending' | 'processing' | 'parsed' | 'completed' | 'error';
    error_message?: string;
    chunk_count?: number;
    collectionId: string;
    user_id: string;
    created_at?: string;
}

export interface Title {
    id: string;
    Title: string;
    status: string;
    traffic_potential_score?: number;
    knowledge_gaps_closed?: boolean;
    priority_score?: number;
    priority_level?: string;
    content_outline?: string;
    user_id: string;
}

export interface ManualAction {
    id: string;
    title: string;
    description: string;
    status: string;
    user_id: string;

    // Enhanced fields
    action_type?: string;
    resource_name?: string;
    estimated_effort_hours?: number;
    difficulty_level?: string; // 'beginner', 'intermediate', 'advanced'
    expected_benefit?: string;
    cost_estimate?: string;
    implementation_notes?: string;
    impact_score?: number;
    priority_level?: string; // 'high', 'medium', 'low', 'critical'

    created_at?: string;
    completed_at?: string | null;
}

export interface KnowledgeGap {
    title_id: string;
    title: string;
    priority_score: number;
    priority_label: string;
    recommended_action: string;
}

export interface GapAnalysisResult {
    gapsFound: number;
    knowledgeGaps: KnowledgeGap[];
    message: string;
    detailedMessage?: string;
    insights?: {
        highPriorityCount: number;
        mediumPriorityCount: number;
        lowPriorityCount: number;
        estimatedTotalTime: number;
        bulkProcessingRecommended: boolean;
        processingStrategy: string;
    };
    enhancedCount?: number;
    closedCount?: number;
}

export interface RagDocumentUsage {
    doc_id?: string;
    title?: string;
    author?: string;
    chunks_contributed?: number;
    importance_weight?: number;
}

export interface RagQueryResponse {
    status: string;
    response?: string;
    error?: string;
    method?: string;
    time_seconds?: number;
    source_attribution?: string[];
    documents_used?: RagDocumentUsage[];
}
