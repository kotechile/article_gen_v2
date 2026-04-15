export type TopicCandidateSource = 'seed' | 'ai' | 'news' | 'manual'

export interface ProjectCategory {
  id: string
  project_id: string
  user_id: string
  name: string
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
  topic_source: TopicCandidateSource
  source_label: string | null
  created_at: string
  updated_at: string
}

