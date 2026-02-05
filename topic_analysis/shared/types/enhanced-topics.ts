/**
 * Enhanced Topics Types
 * TypeScript types for Google Autocomplete integration with topic decomposition
 */

// Enums
export enum SubtopicSource {
  LLM = 'llm',
  AUTOCOMPLETE = 'autocomplete',
  HYBRID = 'hybrid'
}

export enum IndicatorType {
  HIGH_SEARCH_VOLUME = 'high_search_volume',
  TRENDING = 'trending',
  COMMERCIAL_INTENT = 'commercial_intent',
  LOW_COMPETITION = 'low_competition'
}

// Core Models
export interface EnhancedSubtopic {
  id: string;
  title: string;
  search_volume_indicators: string[];
  autocomplete_suggestions: string[];
  relevance_score: number;
  source: SubtopicSource;
  created_at: string;
  updated_at: string;
}

export interface AutocompleteResult {
  query: string;
  suggestions: string[];
  total_suggestions: number;
  processing_time: number;
  timestamp: string;
  success: boolean;
  error_message?: string;
}

export interface SearchVolumeIndicator {
  indicator_type: IndicatorType;
  confidence_level: number;
  description: string;
  source_data: string[];
}

export interface MethodResult {
  subtopics: string[];
  processing_time: number;
  method: string;
}

export interface MethodComparison {
  id: string;
  original_query: string;
  llm_only: MethodResult;
  autocomplete_only: MethodResult;
  hybrid: MethodResult;
  comparison_metrics: Record<string, any>;
  created_at: string;
}

export interface ComparisonMetrics {
  llm_processing_time: number;
  autocomplete_processing_time: number;
  hybrid_processing_time: number;
  llm_relevance_avg: number;
  autocomplete_relevance_avg: number;
  hybrid_relevance_avg: number;
  total_suggestions_found: number;
}

// Request/Response Models
export interface EnhancedTopicDecompositionRequest {
  search_query: string;
  user_id: string;
  max_subtopics?: number;
  use_autocomplete?: boolean;
  use_llm?: boolean;
}

export interface EnhancedTopicDecompositionResponse {
  success: boolean;
  message: string;
  original_query: string;
  subtopics: EnhancedSubtopic[];
  autocomplete_data?: AutocompleteResult;
  processing_time: number;
  enhancement_methods: string[];
}

export interface AutocompleteRequest {
  query: string;
}

export interface AutocompleteResponse {
  success: boolean;
  query: string;
  suggestions: string[];
  total_suggestions: number;
  processing_time: number;
}

export interface MethodComparisonRequest {
  search_query: string;
  user_id: string;
  max_subtopics?: number;
}

export interface MethodComparisonResponse {
  success: boolean;
  original_query: string;
  comparison: {
    llm_only: MethodResult;
    autocomplete_only: MethodResult;
    hybrid: MethodResult;
  };
  recommendation: string;
  processing_time: number;
}

export interface ErrorResponse {
  success: false;
  error: string;
  message: string;
  details?: Record<string, any>;
}

// API Configuration
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  rateLimitDelay: number;
}

export interface CacheConfig {
  ttl: number; // Time to live in seconds
  maxSize: number;
  enabled: boolean;
}

// Service Interfaces
export interface EnhancedTopicsService {
  decomposeTopic(request: EnhancedTopicDecompositionRequest): Promise<EnhancedTopicDecompositionResponse>;
  getAutocompleteSuggestions(query: string): Promise<AutocompleteResponse>;
  compareMethods(request: MethodComparisonRequest): Promise<MethodComparisonResponse>;
  clearCache(): Promise<void>;
  getCacheStats(): Promise<Record<string, any>>;
}

export interface AutocompleteService {
  getSuggestions(query: string): Promise<AutocompleteResponse>;
  getSuggestionsBatch(queries: string[]): Promise<AutocompleteResponse[]>;
  getSuggestionsWithVariations(baseQuery: string): Promise<AutocompleteResponse>;
  clearCache(): Promise<void>;
  getCacheStats(): Promise<Record<string, any>>;
}

// Hook Types
export interface UseEnhancedTopicsOptions {
  enabled?: boolean;
  refetchOnWindowFocus?: boolean;
  staleTime?: number;
  cacheTime?: number;
}

export interface UseEnhancedTopicsReturn {
  decomposeTopic: (request: EnhancedTopicDecompositionRequest) => Promise<EnhancedTopicDecompositionResponse>;
  getAutocompleteSuggestions: (query: string) => Promise<AutocompleteResponse>;
  compareMethods: (request: MethodComparisonRequest) => Promise<MethodComparisonResponse>;
  isLoading: boolean;
  error: Error | null;
  data: EnhancedTopicDecompositionResponse | null;
  clearCache: () => Promise<void>;
}

export interface UseAutocompleteOptions {
  enabled?: boolean;
  debounceMs?: number;
  minQueryLength?: number;
  maxSuggestions?: number;
}

export interface UseAutocompleteReturn {
  suggestions: string[];
  isLoading: boolean;
  error: Error | null;
  getSuggestions: (query: string) => Promise<void>;
  clearSuggestions: () => void;
}

// Component Props
export interface EnhancedTopicDecompositionStepProps {
  onSubtopicsGenerated?: (subtopics: EnhancedSubtopic[]) => void;
  onMethodComparison?: (comparison: MethodComparison) => void;
  initialQuery?: string;
  maxSubtopics?: number;
  showMethodComparison?: boolean;
  showRelevanceScores?: boolean;
  showSearchVolumeIndicators?: boolean;
}

export interface AutocompleteInputProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (suggestion: string) => void;
  placeholder?: string;
  disabled?: boolean;
  maxSuggestions?: number;
  debounceMs?: number;
}

export interface SubtopicCardProps {
  subtopic: EnhancedSubtopic;
  onSelect?: (subtopic: EnhancedSubtopic) => void;
  onDeselect?: (subtopic: EnhancedSubtopic) => void;
  selected?: boolean;
  showRelevanceScore?: boolean;
  showSearchVolumeIndicators?: boolean;
  showAutocompleteSuggestions?: boolean;
}

export interface MethodComparisonProps {
  comparison: MethodComparison;
  onMethodSelect?: (method: string) => void;
  showRecommendation?: boolean;
  showPerformanceMetrics?: boolean;
}

// Utility Types
export type ApiResponse<T> = {
  success: true;
  data: T;
} | {
  success: false;
  error: string;
  message: string;
};

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export type DecompositionMethod = 'llm_only' | 'autocomplete_only' | 'hybrid';

export type RelevanceScore = number; // 0.0 to 1.0

export type ProcessingTime = number; // in seconds

// Event Types
export interface SubtopicSelectedEvent {
  subtopic: EnhancedSubtopic;
  method: DecompositionMethod;
  timestamp: string;
}

export interface MethodComparisonEvent {
  comparison: MethodComparison;
  selectedMethod: string;
  timestamp: string;
}

export interface AutocompleteSuggestionEvent {
  query: string;
  suggestion: string;
  timestamp: string;
}

// Configuration Types
export interface EnhancedTopicsConfig {
  api: ApiConfig;
  cache: CacheConfig;
  autocomplete: {
    enabled: boolean;
    debounceMs: number;
    minQueryLength: number;
    maxSuggestions: number;
  };
  llm: {
    enabled: boolean;
    fallbackEnabled: boolean;
  };
  performance: {
    maxProcessingTime: number;
    timeoutMs: number;
  };
}

// Validation Types
export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

// Analytics Types
export interface AnalyticsEvent {
  event: string;
  properties: Record<string, any>;
  timestamp: string;
  userId: string;
}

export interface PerformanceMetrics {
  decompositionTime: number;
  autocompleteTime: number;
  llmTime: number;
  totalTime: number;
  cacheHitRate: number;
  errorRate: number;
}

