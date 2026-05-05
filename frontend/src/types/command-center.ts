export type TopicCandidateSource = 'seed' | 'ai' | 'news' | 'manual'
export type TopicMode = 'keyword_first' | 'editorial_first' | 'hybrid'
export type KeywordViabilityLabel = 'high' | 'medium' | 'low'

export interface ProjectCategory {
  id: string
  project_id: string
  user_id: string
  name: string
  description?: string | null
  slug: string
  level: number
  parent_category_id: string | null
  sort_order: number
  created_at: string
  updated_at: string
}

export interface TopicCandidate {
  id: string
  project_id: string
  user_id: string
  primary_category_id: string
  secondary_category_id: string | null
  title: string
  rationale?: string | null
  intent_bucket?: string | null
  decision_focus?: string | null
  angle_question?: string | null
  value_layer_tags?: string[] | null
  related_terms?: string[] | null
  source_signals?: string[] | null
  topic_mode?: TopicMode | null
  keyword_viability_score?: number | null
  keyword_viability_label?: KeywordViabilityLabel | null
  topic_generation_reasoning?: string | null
  topic_generation_metadata?: Record<string, any> | null
  topic_source: TopicCandidateSource
  source_label: string | null
  created_at: string
  updated_at: string
}

export interface TopicDraft {
  title: string
  rationale?: string | null
  intent_bucket?: string | null
  decision_focus?: string | null
  angle_question?: string | null
  value_layer_tags?: string[] | null
  related_terms?: string[] | null
  source_signals?: string[] | null
  topic_mode?: TopicMode | null
  keyword_viability_score?: number | null
  keyword_viability_label?: KeywordViabilityLabel | null
  topic_generation_reasoning?: string | null
  topic_generation_metadata?: Record<string, any> | null
}
