import { supabase } from '@/lib/supabase'
import { apiClient } from '@/api-client'
import type { Project } from '@/types'
import type { ProjectCategory, TopicCandidate, TopicCandidateSource, TopicDraft } from '@/types/command-center'
import { researchTopicsService } from './research-topics.service'

interface TopicInsert {
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
  topic_source: TopicCandidateSource
  source_label?: string | null
}

interface StartResearchInput {
  project: Project
  userId: string
  primaryCategory: ProjectCategory
  secondaryCategory: ProjectCategory | null
  topics: TopicCandidate[]
}

type TrendReportTopic = TopicDraft

interface TrendTaskResponse {
  task_id?: string
}

interface TrendTaskStatus {
  status?: string
  result?: {
    status?: string
    error?: string
    result?: {
      report_content?: {
        topics?: TrendReportTopic[]
      }
    }
  }
  error?: string
}

class CommandCenterService {
  private extractTrendTitles(project: Project): string[] {
    const topics = project?.last_trend_report?.report_content?.topics
    if (!Array.isArray(topics)) {
      return []
    }

    return topics
      .map((item) => (typeof item?.title === 'string' ? item.title.trim() : ''))
      .filter(Boolean)
      .slice(0, 8)
  }

  async listCategories(projectId: string): Promise<ProjectCategory[]> {
    const { data, error } = await supabase
      .from('project_categories')
      .select('*')
      .eq('project_id', projectId)
      .order('level', { ascending: true })
      .order('sort_order', { ascending: true })
      .order('name', { ascending: true })

    if (error) {
      throw error
    }

    return (data as ProjectCategory[]) || []
  }

  async listTopicCandidates(projectId: string, secondaryCategoryId: string | null): Promise<TopicCandidate[]> {
    let query = supabase
      .from('project_topic_candidates')
      .select('*')
      .eq('project_id', projectId)
      .order('created_at', { ascending: true })

    if (secondaryCategoryId) {
      query = query.eq('secondary_category_id', secondaryCategoryId)
    }

    const { data, error } = await query

    if (error) {
      throw error
    }

    return (data as TopicCandidate[]) || []
  }

  async createTopicCandidate(input: TopicInsert): Promise<TopicCandidate> {
    const { data, error } = await supabase
      .from('project_topic_candidates')
      .insert([{
        ...input,
        source_label: input.source_label ?? this.getSourceLabel(input.topic_source),
      }])
      .select('*')
      .single()

    if (error) {
      throw error
    }

    return data as TopicCandidate
  }

  async createTopicCandidates(inputs: TopicInsert[]): Promise<TopicCandidate[]> {
    if (!inputs.length) {
      return []
    }

    const payload = inputs.map((input) => ({
      ...input,
      source_label: input.source_label ?? this.getSourceLabel(input.topic_source),
    }))

    const { data, error } = await supabase
      .from('project_topic_candidates')
      .insert(payload)
      .select('*')

    if (error) {
      throw error
    }

    return (data as TopicCandidate[]) || []
  }

  async deleteTopicCandidate(id: string): Promise<void> {
    const { error } = await supabase
      .from('project_topic_candidates')
      .delete()
      .eq('id', id)

    if (error) {
      throw error
    }
  }

  async generateAiTopics(params: {
    project: Project
    primaryCategory: ProjectCategory
    secondaryCategory: ProjectCategory | null
  }): Promise<TopicDraft[]> {
    const response = await apiClient.post<{ topics?: TopicDraft[] }>('/ai/propose-topics', {
      niche_description: this.buildContextDescription(params.project, params.primaryCategory, params.secondaryCategory),
      project_name: params.project.domain || params.project.app_name || '',
      project_description: params.project.site_description || params.project.websiteDescription || params.project.targetAudienceDescription || '',
      primary_category: params.primaryCategory.name,
      primary_category_description: params.primaryCategory.description ?? null,
      secondary_category: params.secondaryCategory?.name ?? null,
      secondary_category_description: params.secondaryCategory?.description ?? null,
      trend_titles: this.extractTrendTitles(params.project),
      count: 20,
    })

    return (response.topics || [])
      .map((topic) => ({
        ...topic,
        title: topic.title?.trim() || '',
      }))
      .filter((topic) => Boolean(topic.title))
  }

  async generateNewsTopics(params: {
    project: Project
    primaryCategory: ProjectCategory
    secondaryCategory: ProjectCategory | null
  }): Promise<TopicDraft[]> {
    const start = await apiClient.post<TrendTaskResponse>(`/v1/trends/${params.project.id}`, {
      primary_category_id: params.primaryCategory.id,
      secondary_category_id: params.secondaryCategory?.id || null,
      project_name: params.project.domain || params.project.app_name || '',
      project_description: params.project.site_description || params.project.websiteDescription || params.project.targetAudienceDescription || '',
      niche_description: this.buildContextDescription(params.project, params.primaryCategory, params.secondaryCategory),
      primary_category: params.primaryCategory.name,
      primary_category_description: params.primaryCategory.description ?? null,
      secondary_category: params.secondaryCategory?.name ?? null,
      secondary_category_description: params.secondaryCategory?.description ?? null,
    })

    if (!start.task_id) {
      throw new Error('Trend analysis could not be started')
    }

    for (let attempt = 0; attempt < 30; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 2000))

      const status = await apiClient.get<TrendTaskStatus>(`/v1/trends/task/${start.task_id}`)

      if (status.status === 'FAILURE') {
        throw new Error(status.error || 'Trend analysis failed')
      }

      if (status.status === 'SUCCESS') {
        const nestedFailure = status.result?.status === 'FAILURE'
        if (nestedFailure) {
          throw new Error(status.result?.error || 'Trend analysis failed')
        }

        const topics = status.result?.result?.report_content?.topics || []
        console.log('Hot News: trend task success', {
          taskId: start.task_id,
          topicsCount: Array.isArray(topics) ? topics.length : 'non_list',
          topicsPreview: Array.isArray(topics) ? topics.slice(0, 5) : topics,
          rawResultKeys: Object.keys(status.result?.result || {}),
        })
        return topics
          .map((topic) => ({
            ...topic,
            title: topic.title?.trim() || '',
          }))
          .filter((topic) => Boolean(topic.title))
      }
    }

    throw new Error('Trend analysis timed out')
  }

  async startResearch(input: StartResearchInput): Promise<Array<{ id: string; title: string }>> {
    const payload = input.topics.map((topic) => ({
      title: topic.title,
      description: this.buildResearchDescription(
        input.project,
        input.primaryCategory,
        input.secondaryCategory,
        topic,
      ),
      project_id: input.project.id,
      primary_category_id: input.primaryCategory.id,
      secondary_category_id: input.secondaryCategory?.id ?? null,
      topic_source: topic.topic_source,
      source_topic_id: topic.id,
      intent_bucket: topic.intent_bucket || null,
      decision_focus: topic.decision_focus || null,
      angle_question: topic.angle_question || null,
      value_layer_tags: topic.value_layer_tags || null,
      target_audience: null,
      evidence_sources: topic.source_signals || null,
      related_terms: topic.related_terms || null,
    }))

    const created = await researchTopicsService.bulkCreateResearchTopics(payload)

    const releasedTopicCandidateIds = input.topics
      .map((topic) => topic.id)
      .filter((id): id is string => Boolean(id))

    if (releasedTopicCandidateIds.length > 0) {
      const { error } = await supabase
        .from('project_topic_candidates')
        .delete()
        .in('id', releasedTopicCandidateIds)

      if (error) {
        // Keep the release successful even if candidate cleanup fails.
        console.warn('Failed to remove released topics from New Research workspace:', error)
      }
    }

    return created.map((topic) => ({ id: topic.id, title: topic.title }))
  }

  getSourceLabel(source: TopicCandidateSource): string {
    switch (source) {
      case 'ai':
        return 'AI Generated'
      case 'news':
        return 'Hot in the News'
      case 'manual':
        return 'Manual Entry'
      default:
        return 'Starter Topic'
    }
  }

  private buildContextDescription(
    project: Project,
    primaryCategory: ProjectCategory,
    secondaryCategory: ProjectCategory | null,
  ): string {
    const label = project.domain || project.app_name || 'this website'
    const description = project.site_description || project.websiteDescription || ''

    return [
      `Website: ${label}`,
      description ? `Website description: ${description}` : null,
      `Primary category: ${primaryCategory.name}`,
      primaryCategory.description ? `Primary category context: ${primaryCategory.description}` : null,
      secondaryCategory ? `Subcategory: ${secondaryCategory.name}` : null,
      secondaryCategory?.description ? `Subcategory context: ${secondaryCategory.description}` : null,
      'Suggest broad topics that are strong starting points for research and article planning.',
    ]
      .filter(Boolean)
      .join('\n')
  }

  private buildResearchDescription(
    project: Project,
    primaryCategory: ProjectCategory,
    secondaryCategory: ProjectCategory | null,
    topic: TopicCandidate,
  ): string {
    const label = project.domain || project.app_name || 'selected website'
    const categoryPath = [primaryCategory.name, secondaryCategory?.name].filter(Boolean).join(' / ')
    if (topic.rationale?.trim()) {
      return topic.rationale.trim()
    }
    return `Research workflow for ${label} in ${categoryPath}: ${topic.title}`
  }
}

export const commandCenterService = new CommandCenterService()
