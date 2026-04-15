import { supabase } from '@/lib/supabase'
import { apiClient } from '@/api-client'
import type { Project } from '@/types'
import type { ProjectCategory, TopicCandidate, TopicCandidateSource } from '@/types/command-center'
import { researchTopicsService } from './research-topics.service'

interface TopicInsert {
  project_id: string
  user_id: string
  primary_category_id: string
  secondary_category_id: string | null
  title: string
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

interface TrendReportTopic {
  title: string
  rationale?: string
}

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
  }): Promise<string[]> {
    const response = await apiClient.post<{ topics?: Array<{ title: string }> }>('/ai/propose-topics', {
      niche_description: this.buildContextDescription(params.project, params.primaryCategory, params.secondaryCategory),
      count: 6,
    })

    return (response.topics || [])
      .map((topic) => topic.title?.trim())
      .filter((title): title is string => Boolean(title))
  }

  async generateNewsTopics(params: {
    project: Project
    primaryCategory: ProjectCategory
    secondaryCategory: ProjectCategory | null
  }): Promise<string[]> {
    const start = await apiClient.post<TrendTaskResponse>(`/v1/trends/${params.project.id}`)

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
        return topics
          .map((topic) => topic.title?.trim())
          .filter((title): title is string => Boolean(title))
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
        topic.title,
      ),
      project_id: input.project.id,
      primary_category_id: input.primaryCategory.id,
      secondary_category_id: input.secondaryCategory?.id ?? null,
      topic_source: topic.topic_source,
      source_topic_id: topic.id,
    }))

    const created = await researchTopicsService.bulkCreateResearchTopics(payload)
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
      secondaryCategory ? `Subcategory: ${secondaryCategory.name}` : null,
      'Suggest broad topics that are strong starting points for research and article planning.',
    ]
      .filter(Boolean)
      .join('\n')
  }

  private buildResearchDescription(
    project: Project,
    primaryCategory: ProjectCategory,
    secondaryCategory: ProjectCategory | null,
    title: string,
  ): string {
    const label = project.domain || project.app_name || 'selected website'
    const categoryPath = [primaryCategory.name, secondaryCategory?.name].filter(Boolean).join(' / ')
    return `Research workflow for ${label} in ${categoryPath}: ${title}`
  }
}

export const commandCenterService = new CommandCenterService()
