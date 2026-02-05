/**
 * Autocomplete Helpers
 * Utility functions for Google Autocomplete integration
 */

import { AutocompleteResult, EnhancedSubtopic, SubtopicSource } from '../types/enhanced-topics';

/**
 * Filter and clean autocomplete suggestions
 */
export function filterSuggestions(suggestions: string[], options: {
  minLength?: number;
  maxLength?: number;
  removeDuplicates?: boolean;
  filterEmpty?: boolean;
} = {}): string[] {
  const {
    minLength = 2,
    maxLength = 100,
    removeDuplicates = true,
    filterEmpty = true
  } = options;

  let filtered = suggestions;

  // Filter by length
  if (minLength > 0 || maxLength < Infinity) {
    filtered = filtered.filter(suggestion => 
      suggestion.length >= minLength && suggestion.length <= maxLength
    );
  }

  // Filter empty suggestions
  if (filterEmpty) {
    filtered = filtered.filter(suggestion => suggestion.trim().length > 0);
  }

  // Remove duplicates
  if (removeDuplicates) {
    const seen = new Set<string>();
    filtered = filtered.filter(suggestion => {
      const normalized = suggestion.toLowerCase().trim();
      if (seen.has(normalized)) {
        return false;
      }
      seen.add(normalized);
      return true;
    });
  }

  return filtered;
}

/**
 * Calculate relevance score for a suggestion
 */
export function calculateRelevanceScore(
  suggestion: string,
  query: string,
  autocompleteData?: AutocompleteResult
): number {
  let score = 0.5; // Base score

  // Boost score if suggestion contains query terms
  const queryTerms = query.toLowerCase().split(' ');
  const suggestionLower = suggestion.toLowerCase();

  for (const term of queryTerms) {
    if (suggestionLower.includes(term)) {
      score += 0.1;
    }
  }

  // Boost score if suggestion appears in autocomplete data
  if (autocompleteData?.success && autocompleteData.suggestions.includes(suggestion)) {
    score += 0.3;
  }

  // Boost score for commercial keywords
  const commercialKeywords = ['best', 'review', 'buy', 'price', 'compare', 'top', 'guide'];
  for (const keyword of commercialKeywords) {
    if (suggestionLower.includes(keyword)) {
      score += 0.05;
    }
  }

  // Boost score for trending indicators
  const trendingIndicators = ['2024', 'new', 'latest', 'trending', 'popular'];
  for (const indicator of trendingIndicators) {
    if (suggestionLower.includes(indicator)) {
      score += 0.05;
    }
  }

  return Math.min(1.0, Math.max(0.0, score));
}

/**
 * Determine the source of a subtopic
 */
export function determineSubtopicSource(
  subtopic: string,
  llmSubtopics: string[],
  autocompleteData?: AutocompleteResult
): SubtopicSource {
  const inLlm = llmSubtopics.includes(subtopic);
  const inAutocomplete = autocompleteData?.success && 
                        autocompleteData.suggestions.includes(subtopic);

  if (inLlm && inAutocomplete) {
    return SubtopicSource.HYBRID;
  } else if (inLlm) {
    return SubtopicSource.LLM;
  } else if (inAutocomplete) {
    return SubtopicSource.AUTOCOMPLETE;
  } else {
    return SubtopicSource.LLM; // Default to LLM
  }
}

/**
 * Create search volume indicators for a subtopic
 */
export function createSearchVolumeIndicators(
  subtopic: string,
  autocompleteData?: AutocompleteResult
): string[] {
  const indicators: string[] = [];

  if (autocompleteData?.success) {
    if (autocompleteData.suggestions.includes(subtopic)) {
      indicators.push('Found in autocomplete suggestions');
    }

    if (autocompleteData.suggestions.length > 5) {
      indicators.push('High search volume from autocomplete');
    }
  }

  // Add generic indicators based on subtopic content
  const subtopicLower = subtopic.toLowerCase();

  if (subtopicLower.includes('best')) {
    indicators.push('High commercial intent');
  }

  if (subtopicLower.includes('2024')) {
    indicators.push('Trending topic');
  }

  if (subtopicLower.includes('review')) {
    indicators.push('Review-focused search');
  }

  return indicators.length > 0 ? indicators : ['Standard search volume'];
}

/**
 * Get autocomplete suggestions related to a subtopic
 */
export function getRelatedAutocompleteSuggestions(
  subtopic: string,
  autocompleteData?: AutocompleteResult,
  maxSuggestions: number = 3
): string[] {
  if (!autocompleteData?.success) {
    return [];
  }

  const relatedSuggestions: string[] = [];
  const subtopicTerms = subtopic.toLowerCase().split(' ');

  for (const suggestion of autocompleteData.suggestions) {
    const suggestionLower = suggestion.toLowerCase();
    if (subtopicTerms.some(term => suggestionLower.includes(term))) {
      relatedSuggestions.push(suggestion);
    }
  }

  return relatedSuggestions.slice(0, maxSuggestions);
}

/**
 * Debounce function for autocomplete requests
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Throttle function for autocomplete requests
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Validate query for autocomplete
 */
export function validateQuery(query: string): {
  isValid: boolean;
  error?: string;
} {
  if (!query || typeof query !== 'string') {
    return { isValid: false, error: 'Query is required' };
  }

  const trimmed = query.trim();
  if (trimmed.length === 0) {
    return { isValid: false, error: 'Query cannot be empty' };
  }

  if (trimmed.length < 1) {
    return { isValid: false, error: 'Query must be at least 1 character' };
  }

  if (trimmed.length > 200) {
    return { isValid: false, error: 'Query must be 200 characters or less' };
  }

  return { isValid: true };
}

/**
 * Format processing time for display
 */
export function formatProcessingTime(timeInSeconds: number): string {
  if (timeInSeconds < 1) {
    return `${Math.round(timeInSeconds * 1000)}ms`;
  } else if (timeInSeconds < 60) {
    return `${timeInSeconds.toFixed(1)}s`;
  } else {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.round(timeInSeconds % 60);
    return `${minutes}m ${seconds}s`;
  }
}

/**
 * Format relevance score for display
 */
export function formatRelevanceScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * Sort suggestions by relevance
 */
export function sortSuggestionsByRelevance(
  suggestions: string[],
  query: string,
  autocompleteData?: AutocompleteResult
): string[] {
  return suggestions.sort((a, b) => {
    const scoreA = calculateRelevanceScore(a, query, autocompleteData);
    const scoreB = calculateRelevanceScore(b, query, autocompleteData);
    return scoreB - scoreA; // Descending order
  });
}

/**
 * Generate query variations for enhanced search
 */
export function generateQueryVariations(baseQuery: string): string[] {
  const variations = [baseQuery];

  // Add common variations
  const commonSuffixes = [' affiliate', ' program', ' marketing', ' review'];
  for (const suffix of commonSuffixes) {
    variations.push(`${baseQuery}${suffix}`);
  }

  // Add prefix variations
  const commonPrefixes = ['best ', 'top ', 'guide to '];
  for (const prefix of commonPrefixes) {
    variations.push(`${prefix}${baseQuery}`);
  }

  return variations;
}

/**
 * Extract keywords from a query
 */
export function extractKeywords(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(word => word.length > 2)
    .filter(word => !isStopWord(word));
}

/**
 * Check if a word is a stop word
 */
function isStopWord(word: string): boolean {
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
    'her', 'us', 'them'
  ]);
  return stopWords.has(word);
}

/**
 * Calculate similarity between two strings
 */
export function calculateSimilarity(str1: string, str2: string): number {
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;

  if (longer.length === 0) {
    return 1.0;
  }

  const editDistance = levenshteinDistance(longer, shorter);
  return (longer.length - editDistance) / longer.length;
}

/**
 * Calculate Levenshtein distance between two strings
 */
function levenshteinDistance(str1: string, str2: string): number {
  const matrix = Array(str2.length + 1).fill(null).map(() => 
    Array(str1.length + 1).fill(null)
  );

  for (let i = 0; i <= str1.length; i++) {
    matrix[0][i] = i;
  }

  for (let j = 0; j <= str2.length; j++) {
    matrix[j][0] = j;
  }

  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,     // deletion
        matrix[j - 1][i] + 1,     // insertion
        matrix[j - 1][i - 1] + indicator // substitution
      );
    }
  }

  return matrix[str2.length][str1.length];
}

/**
 * Group suggestions by similarity
 */
export function groupSuggestionsBySimilarity(
  suggestions: string[],
  threshold: number = 0.7
): string[][] {
  const groups: string[][] = [];
  const used = new Set<number>();

  for (let i = 0; i < suggestions.length; i++) {
    if (used.has(i)) continue;

    const group = [suggestions[i]];
    used.add(i);

    for (let j = i + 1; j < suggestions.length; j++) {
      if (used.has(j)) continue;

      const similarity = calculateSimilarity(suggestions[i], suggestions[j]);
      if (similarity >= threshold) {
        group.push(suggestions[j]);
        used.add(j);
      }
    }

    groups.push(group);
  }

  return groups;
}

/**
 * Create a cache key for autocomplete requests
 */
export function createCacheKey(query: string, userId?: string): string {
  const normalizedQuery = query.toLowerCase().trim();
  return userId ? `${userId}:${normalizedQuery}` : normalizedQuery;
}

/**
 * Check if a cache entry is expired
 */
export function isCacheExpired(timestamp: number, ttlSeconds: number): boolean {
  return Date.now() - timestamp > ttlSeconds * 1000;
}

/**
 * Retry function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      if (attempt === maxRetries) {
        throw lastError;
      }

      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

/**
 * Create a timeout promise
 */
export function createTimeoutPromise<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string = 'Operation timed out'
): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]);
}

/**
 * Batch process items with concurrency control
 */
export async function batchProcess<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 5
): Promise<R[]> {
  const results: R[] = [];
  const executing: Promise<void>[] = [];

  for (const item of items) {
    const promise = processor(item).then(result => {
      results.push(result);
    });

    executing.push(promise);

    if (executing.length >= concurrency) {
      await Promise.race(executing);
      executing.splice(executing.findIndex(p => p === promise), 1);
    }
  }

  await Promise.all(executing);
  return results;
}

