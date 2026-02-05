import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { contentIdeasService, ContentIdea } from '../services/contentIdeasService';
import { ahrefsService, AhrefsKeyword, AhrefsContentIdeasRequest } from '../services/ahrefsService';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Grid,
  Button,
  Chip,
  Paper,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Snackbar,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import {
  Lightbulb as LightbulbIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Bookmark as BookmarkIcon,
  Star as StarIcon,
  ContentCopy as CopyIcon,
  Psychology as PsychologyIcon,
  Close as CloseIcon,
  Upload as UploadIcon,
  Publish as PublishIcon
} from '@mui/icons-material';
import { keywordService } from '../services/keywordService';
import { titlesPublishService, PublishIdeasRequest, PublishIdeasResponse } from '../services/titlesPublishService';

interface IdeaBurstPageProps {
  onGenerateIdeas?: (type: 'seed' | 'ahrefs', data?: any) => void;
  onIdeaSelect?: (idea: any) => void;
  onExport?: (format: string) => void;
  selectedTopicId?: string;
  selectedTopicTitle?: string;
  selectedSubtopics?: string[];
}

interface Idea {
  id: string;
  title: string;
  content_type: string;
  primary_keywords: string[];
  secondary_keywords: string[];
  seo_optimization_score: number;
  traffic_potential_score: number;
  total_search_volume: number;
  average_difficulty: number;
  average_cpc: number;
  optimization_tips: string[];
  content_outline: string[];
  created_at: string;
}

interface IdeaBurstSession {
  id: string;
  user_id: string;
  ideas: Idea[];
  selected_ideas: string[];
  filters: {
    content_type: string;
    min_score: number;
    max_difficulty: number;
  };
  sort_by: string;
  created_at: string;
  updated_at: string;
}

const IdeaBurstPage: React.FC<IdeaBurstPageProps> = ({
  onGenerateIdeas,
  onIdeaSelect,
  onExport,
  selectedTopicId: propSelectedTopicId,
  selectedTopicTitle: propSelectedTopicTitle,
  selectedSubtopics: propSelectedSubtopics
}) => {
  const location = useLocation();
  const { user } = useAuth();
  const navigationState = location.state as { 
    selectedTopicId?: string; 
    selectedTopicTitle?: string; 
    selectedSubtopics?: string[] 
  } | null;

  // Get values from props, navigation state, or sessionStorage fallback
  const getStoredState = () => {
    try {
      const stored = sessionStorage.getItem('ideaBurstState');
      return stored ? JSON.parse(stored) : null;
    } catch (e) {
      console.log('IdeaBurst - Failed to parse sessionStorage:', e);
      return null;
    }
  };

  const storedState = getStoredState();
  const derivedSelectedTopicId = propSelectedTopicId || navigationState?.selectedTopicId || storedState?.selectedTopicId;
  const selectedTopicTitle = propSelectedTopicTitle || navigationState?.selectedTopicTitle || storedState?.selectedTopicTitle;
  // Use local state for selectedSubtopics instead of derived value
  // const selectedSubtopics = propSelectedSubtopics || navigationState?.selectedSubtopics || storedState?.selectedSubtopics || [];

  const [session, setSession] = useState<IdeaBurstSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIdeas, setSelectedIdeas] = useState<Set<string>>(new Set());
  const [showFilters, setShowFilters] = useState(false);
  
  // Advanced filters using AHREFS data
  const [filters, setFilters] = useState({
    minVolume: 0,
    maxVolume: 1000000,
    minDifficulty: 0,
    maxDifficulty: 100,
    minCpc: 0,
    maxCpc: 100,
    minTrafficPotential: 0,
    maxTrafficPotential: 1000000,
    minSeoScore: 0,
    maxSeoScore: 100,
    sortBy: 'score' as 'score' | 'volume' | 'difficulty' | 'cpc' | 'traffic_potential',
    sortOrder: 'desc' as 'asc' | 'desc'
  });
  const [researchTopics, setResearchTopics] = useState<any[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<any>(null);
  const [selectedTopicId, setSelectedTopicId] = useState<string | undefined>(derivedSelectedTopicId);
  const [loadedSubtopics, setLoadedSubtopics] = useState<string[]>([]);
  const [displayedSubtopics, setDisplayedSubtopics] = useState<string[]>([]);
  const [selectedSubtopics, setSelectedSubtopics] = useState<string[]>([]);
  const [topicChangedDirectly, setTopicChangedDirectly] = useState(false);
  const [hasInitializedFromNavigation, setHasInitializedFromNavigation] = useState(false);

  // If no navigation state, show a message to guide the user
  const hasNavigationData = (selectedTopicId || derivedSelectedTopicId) && (displayedSubtopics.length > 0 || loadedSubtopics.length > 0);
  const [existingKeywords, setExistingKeywords] = useState<string[]>([]);
  const [keywordsByTopic, setKeywordsByTopic] = useState<Record<string, string[]>>({});
  const [keywordTextArea, setKeywordTextArea] = useState<string>('');
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [isGeneratingKeywords, setIsGeneratingKeywords] = useState(false);
  
  // Content ideas state
  const [contentIdeas, setContentIdeas] = useState<ContentIdea[]>([]);
  const [blogIdeas, setBlogIdeas] = useState<ContentIdea[]>([]);
  const [softwareIdeas, setSoftwareIdeas] = useState<ContentIdea[]>([]);
  const [isGeneratingIdeas, setIsGeneratingIdeas] = useState(false);
  const [refreshCounter, setRefreshCounter] = useState(0);
  
  // Publish to Titles state
  const [selectedIdeasForPublish, setSelectedIdeasForPublish] = useState<Set<string>>(new Set());
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishDialogOpen, setPublishDialogOpen] = useState(false);
  const [publishResult, setPublishResult] = useState<PublishIdeasResponse | null>(null);
  const [selectedIdeaType, setSelectedIdeaType] = useState<'all' | 'blog' | 'software'>('all');
  
  // AHREFS state
  const [ahrefsFile, setAhrefsFile] = useState<File | null>(null);
  const [ahrefsKeywords, setAhrefsKeywords] = useState<AhrefsKeyword[]>([]);
  const [isUploadingAhrefs, setIsUploadingAhrefs] = useState(false);
  const [ahrefsAnalytics, setAhrefsAnalytics] = useState<any>(null);
  const [showAhrefsUpload, setShowAhrefsUpload] = useState(false);

  // Debug logging (after state declarations)
  console.log('IdeaBurst - Navigation State:', navigationState);
  console.log('IdeaBurst - Stored State:', storedState);
  console.log('IdeaBurst - Selected Subtopics:', selectedSubtopics);
  console.log('IdeaBurst - Selected Topic ID:', selectedTopicId);
  console.log('IdeaBurst - Selected Subtopics Type:', typeof selectedSubtopics);
  console.log('IdeaBurst - Selected Subtopics Length:', selectedSubtopics.length);
  console.log('IdeaBurst - Loaded Subtopics:', loadedSubtopics);
  console.log('IdeaBurst - Loaded Subtopics Length:', loadedSubtopics?.length);
  console.log('IdeaBurst - Location State:', location.state);
  console.log('IdeaBurst - Existing Keywords:', existingKeywords);
  console.log('IdeaBurst - Keyword Text Area:', keywordTextArea);


  // Load research topics
  const loadResearchTopics = async () => {
    try {
      setLoading(true);
      const { supabaseResearchTopicsService } = await import('../services/supabaseResearchTopicsService');
      const response = await supabaseResearchTopicsService.listResearchTopics();
      const topics = Array.isArray(response) ? response : (response?.items || []);
      setResearchTopics(topics);
      
      // Auto-select topic if provided
      if (selectedTopicId) {
        const topic = topics.find(t => t.id === selectedTopicId);
        if (topic) {
          setSelectedTopic(topic);
        }
      }
    } catch (error) {
      console.error('Failed to load research topics:', error);
      setError('Failed to load research topics');
    } finally {
      setLoading(false);
    }
  };

  // Save subtopics to database
  const saveSubtopicsToDatabase = async (subtopics: string[], topicId: string) => {
    if (!user?.id || !topicId || subtopics.length === 0) {
      console.log('IdeaBurst - Cannot save subtopics: missing user, topicId, or subtopics');
      console.log('IdeaBurst - User ID:', user?.id);
      console.log('IdeaBurst - Topic ID:', topicId);
      console.log('IdeaBurst - Subtopics length:', subtopics.length);
      return;
    }

    try {
      console.log('IdeaBurst - Saving subtopics to database:', subtopics, 'for topic:', topicId);
      const { affiliateResearchService } = await import('../services/affiliateResearchService');
      
      // Get the main topic title
      const topic = researchTopics.find(t => t.id === topicId);
      const mainTopic = topic?.title || 'Unknown Topic';
      
      console.log('IdeaBurst - Main topic title:', mainTopic);
      console.log('IdeaBurst - Calling storeSubtopics with:', {
        subtopics,
        userId: user.id,
        mainTopic,
        researchTopicId: topicId
      });
      
      // Save subtopics to database
      await affiliateResearchService.storeSubtopics(subtopics, user.id, mainTopic, topicId);
      console.log('IdeaBurst - Successfully saved subtopics to database');
    } catch (error) {
      console.error('IdeaBurst - Failed to save subtopics to database:', error);
    }
  };

  const handleTopicChange = async (topicId: string) => {
    console.log('IdeaBurst - handleTopicChange called with:', topicId);
    const topic = researchTopics.find(t => t.id === topicId);
    console.log('IdeaBurst - Found topic:', topic);
    setSelectedTopic(topic);
    
    // Update the selectedTopicId state for this component
    // This ensures that selectedTopicId is available for other functions
    setSelectedTopicId(topicId);
    
    // Mark that topic was changed directly (not from navigation)
    setTopicChangedDirectly(true);
    
    // Reset initialization flag so we can load subtopics from database
    setHasInitializedFromNavigation(false);
    
    // Always clear existing content ideas and subtopics when switching topics
    setContentIdeas([]);
    setBlogIdeas([]);
    setSoftwareIdeas([]);
    setDisplayedSubtopics([]); // Clear displayed subtopics immediately
    setLoadedSubtopics([]);    // Clear loaded subtopics immediately
    
    // Show loading state for content ideas
    setIsGeneratingIdeas(true);
    
    // Load existing keywords, subtopics, and content ideas for the new topic
    console.log('IdeaBurst - Loading keywords, subtopics, and content ideas for topic:', topicId);
    await loadExistingKeywords(topicId);
    await loadSubtopicsForTopic(topicId);
    await loadExistingContentIdeas(topicId);
    
    // Clear loading state
    setIsGeneratingIdeas(false);
    console.log('IdeaBurst - Topic change completed for:', topicId);
  };

  // Load subtopics for a topic
  const loadSubtopicsForTopic = async (topicId: string) => {
    try {
      console.log('IdeaBurst - Loading subtopics for topic:', topicId);
      console.log('IdeaBurst - User ID:', user?.id);
      console.log('IdeaBurst - Current displayedSubtopics before loading:', displayedSubtopics);
      
      // Use the affiliate research service to get subtopics
      const { affiliateResearchService } = await import('../services/affiliateResearchService');
      const subtopics = await affiliateResearchService.getSubtopicsForTopic(topicId, user?.id || '');
      
      console.log('IdeaBurst - Raw subtopics from database:', subtopics);
      console.log('IdeaBurst - Subtopics type:', typeof subtopics);
      console.log('IdeaBurst - Subtopics length:', subtopics?.length);
      
      if (subtopics && subtopics.length > 0) {
        console.log('IdeaBurst - Setting subtopics from database for topic:', topicId);
        setLoadedSubtopics(subtopics);
        setDisplayedSubtopics(subtopics);
        setSelectedSubtopics(subtopics); // Auto-select all subtopics
        console.log('IdeaBurst - Updated displayedSubtopics with:', subtopics);
        console.log('IdeaBurst - Auto-selected all subtopics:', subtopics.length);
      } else {
        console.log('IdeaBurst - No subtopics found for topic:', topicId, '- clearing subtopics');
        setLoadedSubtopics([]);
        setDisplayedSubtopics([]);
        console.log('IdeaBurst - Cleared displayedSubtopics');
      }
    } catch (error) {
      console.error('IdeaBurst - Failed to load subtopics:', error);
      setLoadedSubtopics([]);
      setDisplayedSubtopics([]);
    }
  };

  // Load existing keywords for a topic
  const loadExistingKeywords = async (topicId: string) => {
    try {
      // Get user ID from auth context
      const userId = user?.id;
      
      let databaseKeywords: string[] = [];
      
      // Try to load from database if user is logged in
      if (userId) {
        try {
          console.log('IdeaBurst - Loading keywords from database for topic:', topicId, 'user:', userId);
          const keywords = await keywordService.loadExistingKeywords(topicId, userId);
          console.log('IdeaBurst - Database keywords response:', keywords);
          databaseKeywords = keywords.map(k => k.keyword);
          console.log('IdeaBurst - Mapped database keywords:', databaseKeywords);
        } catch (dbError) {
          console.error('IdeaBurst - Failed to load keywords from database:', dbError);
        }
      } else {
        console.log('IdeaBurst - No user ID, skipping database load');
      }
      
      // Get local keywords for this topic
      const localKeywords = keywordsByTopic[topicId] || [];
      
      // Merge database and local keywords, remove duplicates
      const allKeywords = [...new Set([...localKeywords, ...databaseKeywords])];
      
      // Update both the display and the topic-specific storage
      console.log('IdeaBurst - Setting existing keywords to:', allKeywords);
      setExistingKeywords(allKeywords);
      setKeywordsByTopic(prev => ({
        ...prev,
        [topicId]: allKeywords
      }));
      console.log('IdeaBurst - Keywords state updated, length:', allKeywords.length);
    } catch (error) {
      console.error('Failed to load existing keywords:', error);
    }
  };

  // Toggle subtopic selection
  const toggleSubtopicSelection = (subtopic: string) => {
    setSelectedSubtopics(prev => {
      if (prev.includes(subtopic)) {
        return prev.filter(s => s !== subtopic);
      } else {
        return [...prev, subtopic];
      }
    });
  };

  // Select all subtopics
  const selectAllSubtopics = () => {
    setSelectedSubtopics([...displayedSubtopics]);
  };

  // Clear all subtopic selections
  const clearSubtopicSelections = () => {
    setSelectedSubtopics([]);
  };

  // Copy subtopics to clipboard
  const copySubtopicsToClipboard = async () => {
    try {
      const subtopicsText = displayedSubtopics.join(', ');
      await navigator.clipboard.writeText(subtopicsText);
      setSnackbarMessage('Subtopics copied to clipboard!');
      setShowSnackbar(true);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      setSnackbarMessage('Failed to copy to clipboard');
      setShowSnackbar(true);
    }
  };


  // Generate keywords using rule-based approach
  const generateRuleBasedKeywords = () => {
    if (displayedSubtopics.length === 0) {
      setSnackbarMessage('Please select subtopics first');
      setShowSnackbar(true);
      return;
    }

    const keywords: string[] = [];
    const topicLower = selectedTopic?.title?.toLowerCase() || '';
    
    displayedSubtopics.forEach(subtopic => {
      const subtopicLower = subtopic.toLowerCase();
      
      // Generate various keyword variations
      const keywordVariations = [
        `${subtopic} guide`,
        `${subtopic} tips`,
        `best ${subtopic}`,
        `${subtopic} tutorial`,
        `how to ${subtopic}`,
        `${subtopic} for beginners`,
        `${subtopic} techniques`,
        `${subtopic} strategies`,
        `${subtopic} tools`,
        `${subtopic} equipment`,
        `${subtopic} software`,
        `${subtopic} courses`,
        `learn ${subtopic}`,
        `${subtopic} basics`,
        `advanced ${subtopic}`,
        `${subtopic} ideas`,
        `${subtopic} examples`,
        `${subtopic} resources`,
        `${subtopic} reviews`,
        `${subtopic} comparison`
      ];
      
      // Add topic-specific variations
      if (topicLower.includes('photography') || subtopicLower.includes('photo')) {
        keywordVariations.push(
          `${subtopic} camera settings`,
          `${subtopic} lighting`,
          `${subtopic} composition`,
          `${subtopic} editing`,
          `${subtopic} gear`
        );
      } else if (topicLower.includes('travel') || subtopicLower.includes('travel')) {
        keywordVariations.push(
          `${subtopic} destinations`,
          `${subtopic} planning`,
          `${subtopic} budget`,
          `${subtopic} itinerary`,
          `${subtopic} tips`
        );
      } else if (topicLower.includes('business') || subtopicLower.includes('marketing')) {
        keywordVariations.push(
          `${subtopic} strategy`,
          `${subtopic} tools`,
          `${subtopic} software`,
          `${subtopic} automation`,
          `${subtopic} analytics`
        );
      }
      
      keywords.push(...keywordVariations);
    });
    
    // Remove duplicates while preserving order
    const uniqueKeywords = Array.from(new Set(keywords));
    
    // Add rule-based keywords to the text area
    const newKeywords = uniqueKeywords.slice(0, 20).join('\n'); // Limit to 20 keywords
    const currentKeywords = keywordTextArea.trim();
    const combinedKeywords = currentKeywords ? `${currentKeywords}\n${newKeywords}` : newKeywords;
    setKeywordTextArea(combinedKeywords);
    
    setSnackbarMessage(`Added ${Math.min(20, uniqueKeywords.length)} rule-based keywords to your list`);
    setShowSnackbar(true);
  };

  // Generate keywords using LLM
  const generateKeywordsWithLLM = async () => {
    try {
      setIsGeneratingKeywords(true);
      setError(null);
      
      const response = await keywordService.generateKeywordsWithLLM({
        subtopics: Array.from(selectedSubtopics),
        topicId: selectedTopic?.id || '',
        topicTitle: selectedTopic?.title || ''
      });

      if (response.success && response.keywords.length > 0) {
        // Add LLM-generated keywords to the text area
        const newKeywords = response.keywords.join('\n');
        const currentKeywords = keywordTextArea.trim();
        const combinedKeywords = currentKeywords ? `${currentKeywords}\n${newKeywords}` : newKeywords;
        setKeywordTextArea(combinedKeywords);
        
        setSnackbarMessage(`Added ${response.keywords.length} LLM-generated keywords to your list`);
        setShowSnackbar(true);
      } else {
        throw new Error(response.message || 'Failed to generate keywords');
      }
    } catch (error) {
      console.error('Failed to generate keywords:', error);
      setError('Failed to generate keywords. Please try again.');
      setSnackbarMessage('Failed to generate keywords. Please try again.');
      setShowSnackbar(true);
    } finally {
      setIsGeneratingKeywords(false);
    }
  };

  // Parse keywords from text area
  const parseKeywordsFromTextArea = (): string[] => {
    if (!keywordTextArea.trim()) return [];
    
    const keywords = keywordTextArea
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .filter((keyword, index, array) => array.indexOf(keyword) === index); // Remove duplicates
    
    console.log('Parsed keywords from text area:', keywords);
    return keywords;
  };

  // Save keywords from text area
  const saveKeywordsFromTextArea = async () => {
    const keywords = parseKeywordsFromTextArea();
    
    if (keywords.length === 0) {
      setSnackbarMessage('No keywords to save');
      setShowSnackbar(true);
      return;
    }

    try {
      setLoading(true);
      const userId = user?.id;
      
      // If no user ID, show error message
      if (!userId) {
        console.log('No user ID found, using anonymous mode');
        setSnackbarMessage('No user ID found - keywords will be displayed locally only');
        setShowSnackbar(true);
        // Don't return, continue with local display
      } else {
        console.log('User ID found:', userId);
      }
      
      // Update local state first (regardless of database save)
      console.log('Saving keywords:', keywords);
      const topicId = selectedTopic?.id;
      
      setExistingKeywords(prev => {
        const newKeywords = [...prev, ...keywords];
        console.log('Updated existing keywords:', newKeywords);
        return newKeywords;
      });
      
      // Also save to topic-specific storage
      if (topicId) {
        setKeywordsByTopic(prev => ({
          ...prev,
          [topicId]: [...(prev[topicId] || []), ...keywords]
        }));
        console.log('Saved keywords to topic-specific storage for topic:', topicId);
      }
      
      // Try to save keywords to database (only if we have a user ID)
      if (userId) {
        try {
          await keywordService.saveKeywords(
            keywords,
            selectedTopic?.id || '',
            userId,
            'manual'
          );
          setSnackbarMessage(`Saved ${keywords.length} keywords successfully!`);
        } catch (dbError) {
          console.error('Database save failed, but keywords are displayed locally:', dbError);
          setSnackbarMessage(`Added ${keywords.length} keywords locally (database save failed)`);
        }
      } else {
        setSnackbarMessage(`Added ${keywords.length} keywords locally (no user ID for database save)`);
      }
      
      setShowSnackbar(true);
      
      // Clear the text area after saving
      setKeywordTextArea('');
    } catch (error) {
      console.error('Failed to save keywords:', error);
      setError('Failed to save keywords. Please try again.');
      setSnackbarMessage('Failed to save keywords. Please try again.');
      setShowSnackbar(true);
    } finally {
      setLoading(false);
    }
  };

  // Remove a keyword from the saved list
  const removeKeyword = (keywordToRemove: string) => {
    setExistingKeywords(prev => prev.filter(keyword => keyword !== keywordToRemove));
    
    // Also remove from topic-specific storage
    const topicId = selectedTopic?.id;
    if (topicId) {
      setKeywordsByTopic(prev => ({
        ...prev,
        [topicId]: (prev[topicId] || []).filter(keyword => keyword !== keywordToRemove)
      }));
    }
    
    setSnackbarMessage(`Removed "${keywordToRemove}" from saved keywords`);
    setShowSnackbar(true);
  };

  // Content Ideas Functions
  const generateContentIdeas = async () => {
    if (!selectedTopicId || !user?.id || selectedSubtopics.length === 0 || existingKeywords.length === 0) {
      setSnackbarMessage('Please select a topic, subtopics, and add some keywords first');
      setShowSnackbar(true);
      return;
    }

    setIsGeneratingIdeas(true);
    setError(null);

    // Clear existing ideas before generating new ones
    setContentIdeas([]);
    setBlogIdeas([]);
    setSoftwareIdeas([]);

    try {
      console.log('🚀 Starting content idea generation with LLM...');
      const response = await contentIdeasService.generateContentIdeas({
        topic_id: selectedTopicId,
        topic_title: selectedTopicTitle || 'photography', // Use a better default
        subtopics: selectedSubtopics,
        keywords: existingKeywords,
        user_id: user.id,
        content_types: ['blog', 'software']
      });

      if (response.success) {
        console.log('✅ Content ideas generated successfully:', response.total_ideas);
        setContentIdeas(response.ideas);
        setBlogIdeas(response.ideas.filter(idea => idea.content_type === 'blog'));
        setSoftwareIdeas(response.ideas.filter(idea => idea.content_type === 'software'));
        
        setSnackbarMessage(`Generated ${response.total_ideas} content ideas (${response.blog_ideas} blog, ${response.software_ideas} software)`);
        setShowSnackbar(true);
      } else {
        console.error('❌ Content idea generation failed:', response);
        setError('Failed to generate content ideas');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate content ideas');
    } finally {
      setIsGeneratingIdeas(false);
    }
  };

  // AHREFS functionality
  const handleAhrefsFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setAhrefsFile(file);
    setIsUploadingAhrefs(true);
    setError(null);

    try {
      // Parse the file locally first
      const text = await file.text();
      const keywords = ahrefsService.parseAhrefsCSV(text);
      
      // Upload to backend
      const uploadResult = await ahrefsService.uploadAhrefsFile(file, selectedTopicId || 'test-topic-123', user?.id || 'ed9fdb80-58db-44fc-8869-1d835408a65f');
      
      if (uploadResult.success) {
        // Use keywords returned by backend if available, otherwise use locally parsed ones
        const backendKeywords = uploadResult.ahrefs_keywords || uploadResult.keywords || [];
        const finalKeywords = backendKeywords.length > 0 ? backendKeywords : keywords;
        
        console.log('Keywords source:', {
          backendCount: backendKeywords.length,
          localCount: keywords.length,
          usingBackend: backendKeywords.length > 0
        });
        
        setAhrefsKeywords(finalKeywords);
        setAhrefsAnalytics(ahrefsService.calculateAnalyticsSummary(finalKeywords));
        
        setSnackbarMessage(`Successfully uploaded and parsed ${finalKeywords.length} keywords from AHREFS file. Generating content ideas...`);
        setShowSnackbar(true);
        
        // Auto-generate content ideas after successful upload
        console.log('Auto-generation check:', {
          selectedTopicId,
          userId: user?.id,
          selectedSubtopics: selectedSubtopics.length || 0,
          displayedSubtopics: displayedSubtopics.length,
          canGenerate: !!(selectedTopicId && user?.id && (selectedSubtopics.length > 0 || displayedSubtopics.length > 0))
        });
        
        // If no subtopics are available, try to load them from the database
        let finalSubtopics = selectedSubtopics.length > 0 ? selectedSubtopics : displayedSubtopics;
        
        if (selectedTopicId && user?.id && finalSubtopics.length === 0) {
          console.log('No subtopics available, trying to load from database...');
          try {
            await loadSubtopicsForTopic(selectedTopicId);
            // Wait a bit for the state to update
            await new Promise(resolve => setTimeout(resolve, 1000));
            // Get the updated subtopics from the state
            finalSubtopics = displayedSubtopics;
            console.log('Subtopics loaded after delay:', finalSubtopics.length);
          } catch (error) {
            console.error('Failed to load subtopics:', error);
          }
        }
        
        if (selectedTopicId && user?.id && finalSubtopics.length > 0) {
          try {
            console.log('Starting auto-generation with AHREFS data...');
            console.log('Using subtopics for generation:', finalSubtopics);
            setIsGeneratingIdeas(true);
            
            // Create the request with the loaded subtopics
            const request: AhrefsContentIdeasRequest = {
              topic_id: selectedTopicId,
              topic_title: selectedTopicTitle || 'Unknown Topic',
              subtopics: finalSubtopics,
              ahrefs_keywords: finalKeywords,
              user_id: user.id
            };

            const response = await ahrefsService.generateContentIdeasWithAhrefs(request);

            if (response.success) {
              setContentIdeas(response.ideas);
              setBlogIdeas(response.ideas.filter(idea => idea.content_type === 'blog'));
              setSoftwareIdeas(response.ideas.filter(idea => idea.content_type === 'software'));
              
              setSnackbarMessage(`Successfully generated content ideas using uploaded AHREFS keywords!`);
              setShowSnackbar(true);
              console.log('Auto-generation completed successfully');
            } else {
              setError('Failed to generate content ideas with AHREFS data');
            }
          } catch (error) {
            console.error('Error generating ideas after upload:', error);
            setSnackbarMessage(`Upload successful, but failed to generate ideas. Please try again.`);
            setShowSnackbar(true);
          } finally {
            setIsGeneratingIdeas(false);
          }
        } else {
          console.log('Auto-generation skipped - missing requirements');
          setSnackbarMessage(`Upload successful! Please select a topic and subtopics to generate content ideas.`);
          setShowSnackbar(true);
        }
      } else {
        throw new Error(uploadResult.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Error uploading AHREFS file:', error);
      setError('Failed to upload AHREFS file. Please check the format and try again.');
    } finally {
      setIsUploadingAhrefs(false);
    }
  };

  const generateContentIdeasWithAhrefs = async () => {
    const subtopicsToUse = selectedSubtopics.length > 0 ? selectedSubtopics : displayedSubtopics;
    
    if (!selectedTopicId || !user?.id || subtopicsToUse.length === 0 || ahrefsKeywords.length === 0) {
      setSnackbarMessage('Please select a topic, subtopics, and upload AHREFS data first');
      setShowSnackbar(true);
      return;
    }

    setIsGeneratingIdeas(true);
    setError(null);

    try {
      const request: AhrefsContentIdeasRequest = {
        topic_id: selectedTopicId,
        topic_title: selectedTopicTitle || 'Unknown Topic',
        subtopics: subtopicsToUse,
        ahrefs_keywords: ahrefsKeywords,
        user_id: user.id
      };

      const response = await ahrefsService.generateContentIdeasWithAhrefs(request);

      if (response.success) {
        setContentIdeas(response.ideas);
        setBlogIdeas(response.ideas.filter(idea => idea.content_type === 'blog'));
        setSoftwareIdeas(response.ideas.filter(idea => idea.content_type === 'software'));
        
        setSnackbarMessage(`Generated ${response.total_ideas} content ideas with AHREFS data (${response.blog_ideas} blog, ${response.software_ideas} software)`);
        setShowSnackbar(true);
      } else {
        setError('Failed to generate content ideas with AHREFS data');
      }
    } catch (error) {
      console.error('Error generating content ideas with AHREFS:', error);
      setError('Failed to generate content ideas with AHREFS data');
    } finally {
      setIsGeneratingIdeas(false);
    }
  };

  const loadExistingContentIdeas = async (topicId?: string) => {
    const currentTopicId = topicId || selectedTopicId;
    if (!currentTopicId || !user?.id) return;

    try {
      const ideas = await contentIdeasService.getContentIdeas(currentTopicId, user.id);
      setContentIdeas(ideas);
      setBlogIdeas(ideas.filter(idea => idea.content_type === 'blog'));
      setSoftwareIdeas(ideas.filter(idea => idea.content_type === 'software'));
    } catch (err) {
      console.error('Failed to load existing content ideas:', err);
    }
  };

  const deleteContentIdea = async (ideaId: string) => {
    if (!user?.id) return;

    try {
      const success = await contentIdeasService.deleteContentIdea(ideaId, user.id);
      if (success) {
        setContentIdeas(prev => prev.filter(idea => idea.id !== ideaId));
        setBlogIdeas(prev => prev.filter(idea => idea.id !== ideaId));
        setSoftwareIdeas(prev => prev.filter(idea => idea.id !== ideaId));
        setSnackbarMessage('Content idea deleted successfully');
        setShowSnackbar(true);
      }
    } catch (err) {
      console.error('Failed to delete content idea:', err);
    }
  };


  useEffect(() => {
    loadResearchTopics();
  }, []);


  // Load keywords when selectedTopicId changes
  useEffect(() => {
    if (selectedTopicId) {
      console.log('Loading keywords for selected topic:', selectedTopicId);
      loadExistingKeywords(selectedTopicId);
      loadExistingContentIdeas();
    }
  }, [selectedTopicId]);

  // Load subtopics when selectedTopicId changes
  useEffect(() => {
    if (selectedTopicId && user?.id) {
      console.log('Loading subtopics for selected topic:', selectedTopicId);
      loadSubtopicsForTopic(selectedTopicId);
    }
  }, [selectedTopicId, user?.id]);

  // Initialize subtopics from navigation state only once
  useEffect(() => {
    if (selectedSubtopics && selectedSubtopics.length > 0 && user?.id && !hasInitializedFromNavigation) {
      // Use the original topic ID from navigation state, not the current selectedTopicId
      const originalTopicId = navigationState?.selectedTopicId || propSelectedTopicId;
      
      if (originalTopicId) {
        console.log('IdeaBurst - Initializing from navigation state:', selectedSubtopics);
        console.log('IdeaBurst - Using original topic ID from navigation:', originalTopicId);
        setLoadedSubtopics(selectedSubtopics);
        setDisplayedSubtopics(selectedSubtopics);
        setHasInitializedFromNavigation(true);
        
        // Save subtopics to database with the original topic ID
        saveSubtopicsToDatabase(selectedSubtopics, originalTopicId);
      }
    }
  }, [selectedSubtopics, user?.id, hasInitializedFromNavigation, navigationState?.selectedTopicId, propSelectedTopicId]);

  // Debug effect to log navigation state changes
  useEffect(() => {
    console.log('IdeaBurst - Navigation state changed:', navigationState);
    console.log('IdeaBurst - Selected subtopics from navigation:', navigationState?.selectedSubtopics);
    console.log('IdeaBurst - Selected topic from navigation:', navigationState?.selectedTopicId);
  }, [navigationState]);

  useEffect(() => {
    // Initialize session with empty data
    setSession({
      id: 'mock-session-id',
      user_id: 'mock-user-id',
      ideas: [],
      selected_ideas: [],
      filters: {
        content_type: 'all',
        min_score: 0,
        max_difficulty: 100
      },
      sort_by: 'score',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    });
  }, []);

  const handleGenerateIdeas = async (type: 'seed' | 'ahrefs') => {
    setLoading(true);
    setError(null);
    
    try {
      if (type === 'seed') {
        await generateContentIdeas();
      } else if (type === 'ahrefs') {
        await generateContentIdeasWithAhrefs();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate ideas');
    } finally {
      setLoading(false);
    }
  };

  const handleIdeaSelect = (idea: Idea) => {
    const newSelected = new Set(selectedIdeas);
    if (newSelected.has(idea.id)) {
      newSelected.delete(idea.id);
    } else {
      newSelected.add(idea.id);
    }
    setSelectedIdeas(newSelected);
    onIdeaSelect?.(idea);
  };

  // Publish to Titles functions
  const handleIdeaCheckboxChange = (ideaId: string, checked: boolean) => {
    const newSelected = new Set(selectedIdeasForPublish);
    if (checked) {
      newSelected.add(ideaId);
    } else {
      newSelected.delete(ideaId);
    }
    setSelectedIdeasForPublish(newSelected);
  };

  const handleSelectAllIdeas = () => {
    const currentIdeas = selectedIdeaType === 'all' ? contentIdeas : 
      selectedIdeaType === 'blog' ? blogIdeas : softwareIdeas;
    // Only select publishable (non-software, non-published) ideas
    const publishableIdeas = currentIdeas.filter(idea => idea.content_type !== 'software' && !idea.published);
    const allIdeaIds = new Set(publishableIdeas.map(idea => idea.id));
    setSelectedIdeasForPublish(allIdeaIds);
  };

  const handleDeselectAllIdeas = () => {
    setSelectedIdeasForPublish(new Set());
  };

  const handlePublishToTitles = async () => {
    if (!user?.id) {
      setSnackbarMessage('You must be logged in to publish ideas');
      setShowSnackbar(true);
      return;
    }

    if (selectedIdeasForPublish.size === 0) {
      setSnackbarMessage('Please select at least one idea to publish');
      setShowSnackbar(true);
      return;
    }

    setIsPublishing(true);
    setPublishDialogOpen(true);

    try {
      const currentIdeas = selectedIdeaType === 'all' ? contentIdeas : 
        selectedIdeaType === 'blog' ? blogIdeas : softwareIdeas;
      // Only publish non-software ideas
      const publishableIdeas = currentIdeas.filter(idea => idea.content_type !== 'software');
      const ideasToPublish = publishableIdeas.filter(idea => selectedIdeasForPublish.has(idea.id));
      
      const publishRequest: PublishIdeasRequest = {
        ideas: ideasToPublish,
        trend_analysis_id: selectedTopicId,
        source_topic_id: selectedTopicId,
        user_id: user.id
      };

      const result = await titlesPublishService.publishIdeas(publishRequest);
      setPublishResult(result);

      if (result.success) {
        setSnackbarMessage(`Successfully published ${result.published_count} ideas to Titles table`);
        setShowSnackbar(true);
        // Clear selected ideas after successful publish
        setSelectedIdeasForPublish(new Set());
        
        // Refresh the content ideas to show updated published status
        try {
          const currentTopicId = derivedSelectedTopicId;
          if (currentTopicId && user?.id) {
            // Add a small delay to ensure database updates are committed
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const refreshedIdeas = await contentIdeasService.getContentIdeas(currentTopicId, user.id);
            console.log('Refreshed ideas after publishing:', refreshedIdeas.length);
            console.log('Published ideas after refresh:', refreshedIdeas.filter(idea => idea.published).length);
            // Force a complete state reset
            setContentIdeas([]);
            setBlogIdeas([]);
            setSoftwareIdeas([]);
            
            // Then set the new data
            setTimeout(() => {
              setContentIdeas(refreshedIdeas);
              setBlogIdeas(refreshedIdeas.filter(idea => idea.content_type === 'blog'));
              setSoftwareIdeas(refreshedIdeas.filter(idea => idea.content_type === 'software'));
              setRefreshCounter(prev => prev + 1);
              console.log('Refreshed content ideas after publishing');
            }, 100);
          }
        } catch (refreshError) {
          console.warn('Failed to refresh content ideas after publishing:', refreshError);
        }
      } else {
        setSnackbarMessage(`Failed to publish ideas. ${result.errors.join(', ')}`);
        setShowSnackbar(true);
      }
    } catch (error: any) {
      console.error('Error publishing ideas:', error);
      setSnackbarMessage(`Error publishing ideas: ${error.message}`);
      setShowSnackbar(true);
    } finally {
      setIsPublishing(false);
    }
  };

  const handlePublishSingleIdea = async (idea: any) => {
    if (!user?.id) {
      setSnackbarMessage('You must be logged in to publish ideas');
      setShowSnackbar(true);
      return;
    }

    if (idea.content_type === 'software') {
      setSnackbarMessage('Software ideas cannot be published to Content Generation');
      setShowSnackbar(true);
      return;
    }

    setIsPublishing(true);
    setPublishDialogOpen(true);

    try {
      const publishRequest: PublishIdeasRequest = {
        ideas: [idea],
        trend_analysis_id: selectedTopicId,
        source_topic_id: selectedTopicId,
        user_id: user.id
      };

      const result = await titlesPublishService.publishIdeas(publishRequest);
      setPublishResult(result);

      if (result.success) {
        setSnackbarMessage(`Successfully published "${idea.title}" to Titles table`);
        setShowSnackbar(true);
        // Remove from selected ideas if it was selected
        const newSelected = new Set(selectedIdeasForPublish);
        newSelected.delete(idea.id);
        setSelectedIdeasForPublish(newSelected);
        
        // Refresh the content ideas to show updated published status
        try {
          const currentTopicId = derivedSelectedTopicId;
          if (currentTopicId && user?.id) {
            // Add a small delay to ensure database updates are committed
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const refreshedIdeas = await contentIdeasService.getContentIdeas(currentTopicId, user.id);
            console.log('Refreshed ideas after single publish:', refreshedIdeas.length);
            console.log('Published ideas after single refresh:', refreshedIdeas.filter(idea => idea.published).length);
            // Force a complete state reset
            setContentIdeas([]);
            setBlogIdeas([]);
            setSoftwareIdeas([]);
            
            // Then set the new data
            setTimeout(() => {
              setContentIdeas(refreshedIdeas);
              setBlogIdeas(refreshedIdeas.filter(idea => idea.content_type === 'blog'));
              setSoftwareIdeas(refreshedIdeas.filter(idea => idea.content_type === 'software'));
              setRefreshCounter(prev => prev + 1);
              console.log('Refreshed content ideas after single publish');
            }, 100);
          }
        } catch (refreshError) {
          console.warn('Failed to refresh content ideas after single publish:', refreshError);
        }
      } else {
        setSnackbarMessage(`Failed to publish "${idea.title}". ${result.errors.join(', ')}`);
        setShowSnackbar(true);
      }
    } catch (error: any) {
      console.error('Error publishing single idea:', error);
      setSnackbarMessage(`Error publishing "${idea.title}": ${error.message}`);
      setShowSnackbar(true);
    } finally {
      setIsPublishing(false);
    }
  };

  // const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
  //   setActiveTab(newValue);
  // };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };

  const getContentTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'article': return 'primary';
      case 'comparison': return 'secondary';
      case 'guide': return 'success';
      case 'tutorial': return 'info';
      case 'review': return 'warning';
      default: return 'default';
    }
  };

  // Get all ideas based on selected type
  const allIdeas = selectedIdeaType === 'all' ? contentIdeas : 
                   selectedIdeaType === 'blog' ? blogIdeas : softwareIdeas;

  const filteredIdeas = allIdeas.filter(idea => {
    // Basic content type filter
    const sessionFilters = session?.filters || { content_type: 'all', min_score: 0, max_difficulty: 100 };
    const contentTypeMatch = sessionFilters.content_type === 'all' || idea.content_type === sessionFilters.content_type;
    
    // AHREFS-based filters
    const volumeMatch = idea.total_search_volume >= filters.minVolume && idea.total_search_volume <= filters.maxVolume;
    const difficultyMatch = idea.average_difficulty >= filters.minDifficulty && idea.average_difficulty <= filters.maxDifficulty;
    const cpcMatch = (idea.average_cpc || 0) >= filters.minCpc && (idea.average_cpc || 0) <= filters.maxCpc;
    const trafficMatch = (idea.traffic_potential_score || 0) >= filters.minTrafficPotential && (idea.traffic_potential_score || 0) <= filters.maxTrafficPotential;
    const seoScoreMatch = (idea.seo_optimization_score || 0) >= filters.minSeoScore && (idea.seo_optimization_score || 0) <= filters.maxSeoScore;
    
    return contentTypeMatch && volumeMatch && difficultyMatch && cpcMatch && trafficMatch && seoScoreMatch;
  }) || [];

  const sortedIdeas = filteredIdeas.sort((a, b) => {
    let comparison = 0;
    
    switch (filters.sortBy) {
      case 'score':
        const scoreA = (a.seo_optimization_score || 0) + (a.traffic_potential_score || 0);
        const scoreB = (b.seo_optimization_score || 0) + (b.traffic_potential_score || 0);
        comparison = scoreB - scoreA;
        break;
      case 'volume':
        comparison = (b.total_search_volume || 0) - (a.total_search_volume || 0);
        break;
      case 'difficulty':
        comparison = (a.average_difficulty || 0) - (b.average_difficulty || 0);
        break;
      case 'cpc':
        comparison = (b.average_cpc || 0) - (a.average_cpc || 0);
        break;
      case 'traffic_potential':
        comparison = (b.traffic_potential_score || 0) - (a.traffic_potential_score || 0);
        break;
      default:
        comparison = 0;
    }
    
    return filters.sortOrder === 'asc' ? -comparison : comparison;
  });

  // Debug logging for ideas
  console.log('IdeaBurst - Debug ideas state:', {
    contentIdeas: contentIdeas.length,
    blogIdeas: blogIdeas.length,
    softwareIdeas: softwareIdeas.length,
    publishedIdeas: contentIdeas.filter(idea => idea.published).length,
    user: user ? { id: user.id, email: user.email } : null,
    isAuthenticated: !!user,
    selectedTopicId: selectedTopicId,
    sampleIdea: contentIdeas[0] ? {
      id: contentIdeas[0].id,
      title: contentIdeas[0].title,
      published: contentIdeas[0].published,
      published_at: contentIdeas[0].published_at,
      published_to_titles: contentIdeas[0].published_to_titles,
      status: contentIdeas[0].status,
      publishedType: typeof contentIdeas[0].published
    } : null,
    selectedIdeaType,
    allIdeas: allIdeas.length,
    filteredIdeas: filteredIdeas.length,
    sortedIdeas: sortedIdeas.length,
    isGeneratingIdeas
  });

  // Additional debug logging for authentication
  if (!user) {
    console.warn('IdeaBurst - No user authenticated. This will prevent content ideas from loading.');
  } else {
    console.log('IdeaBurst - User authenticated:', {
      id: user.id,
      email: user.email,
      name: user.name
    });
  }

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Generating ideas...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Idea Burst
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Navigation Guidance */}
      {!hasNavigationData && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>No subtopics selected.</strong> To use Idea Burst effectively, please:
            <br />
            1. Go to the <strong>Trend Analysis</strong> tab
            <br />
            2. Select a research topic and generate subtopics
            <br />
            3. Select multiple subtopics using the checkboxes
            <br />
            4. Click "Go to Idea Burst" to continue with your selected subtopics
            <br />
            <br />
            <strong>Note:</strong> You can still use the keyword generation tools below, but they work best with selected subtopics.
          </Typography>
        </Alert>
      )}

      {/* Research Topic Selection */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Research Topic
        </Typography>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Select Research Topic</InputLabel>
          <Select
            value={selectedTopic?.id || ''}
            onChange={(e) => handleTopicChange(e.target.value)}
            label="Select Research Topic"
            disabled={loading}
          >
            {researchTopics.map((topic) => (
              <MenuItem key={topic.id} value={topic.id}>
                {topic.title}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        {selectedTopic && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Selected: {selectedTopic.title}
            </Typography>
            {navigationState && (
              <Chip
                label="Pre-selected from Trend Analysis"
                color="info"
                size="small"
              />
            )}
          </Box>
        )}

        {/* Available Subtopics */}
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Available Subtopics ({displayedSubtopics.length})
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant="outlined"
                onClick={selectAllSubtopics}
                disabled={displayedSubtopics.length === 0}
              >
                Select All
              </Button>
              <Button
                size="small"
                variant="outlined"
                onClick={clearSubtopicSelections}
                disabled={selectedSubtopics.length === 0}
              >
                Clear All
              </Button>
            </Box>
          </Box>
          
          {displayedSubtopics.length > 0 ? (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {displayedSubtopics.map((subtopic, index) => (
                <Chip
                  key={`subtopic-${index}-${subtopic}`}
                  label={subtopic}
                  color={selectedSubtopics.includes(subtopic) ? "primary" : "default"}
                  variant={selectedSubtopics.includes(subtopic) ? "filled" : "outlined"}
                  onClick={() => toggleSubtopicSelection(subtopic)}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No subtopics available. Please select a topic first.
            </Typography>
          )}
          
          {selectedSubtopics.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="primary">
                Selected: {selectedSubtopics.length} subtopic{selectedSubtopics.length !== 1 ? 's' : ''}
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>

      {/* Keyword Management Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Keyword Management
        </Typography>
          
          {/* Action Buttons */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
            <Button
              variant="outlined"
              startIcon={<CopyIcon />}
              onClick={copySubtopicsToClipboard}
              disabled={displayedSubtopics.length === 0}
            >
              Copy Subtopics to Clipboard
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<PsychologyIcon />}
              onClick={generateRuleBasedKeywords}
              disabled={displayedSubtopics.length === 0}
            >
              Generate Rule-Based Keywords
            </Button>
            
            <Button
              variant="contained"
              startIcon={<PsychologyIcon />}
              onClick={generateKeywordsWithLLM}
              disabled={displayedSubtopics.length === 0 || isGeneratingKeywords}
            >
              {isGeneratingKeywords ? 'Generating...' : 'Generate Keywords with LLM'}
            </Button>
          </Box>

          {/* Main Keyword Text Area */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Build Your Keyword List
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Use the buttons above to add keywords, or type/paste them manually (one per line). 
              You can paste keywords from any source (Ahrefs, Google Keyword Planner, etc.) directly into this text area.
              When you're happy with your list, click "Save Keywords" to store them.
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={8}
              value={keywordTextArea}
              onChange={(e) => setKeywordTextArea(e.target.value)}
              placeholder="Enter or paste keywords here, one per line...&#10;You can paste from Ahrefs, Google Keyword Planner, or any other source.&#10;Use the buttons above to generate keywords automatically."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={saveKeywordsFromTextArea}
                disabled={!keywordTextArea.trim() || loading}
              >
                Save Keywords ({parseKeywordsFromTextArea().length})
              </Button>
              <Button
                variant="outlined"
                onClick={() => setKeywordTextArea('')}
                disabled={!keywordTextArea.trim()}
              >
                Clear List
              </Button>
            </Box>
          </Box>


          {/* Saved Keywords Display */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Saved Keywords {existingKeywords.length > 0 && `(${existingKeywords.length})`}
            </Typography>
            
            {/* DEBUG: Show keywords count and first few keywords */}
            <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary">
                DEBUG: Keywords count: {existingKeywords.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                DEBUG: First 3 keywords: {existingKeywords.slice(0, 3).join(', ')}
              </Typography>
            </Box>
            
            {console.log('IdeaBurst - Rendering keywords section, existingKeywords.length:', existingKeywords.length)}
            {existingKeywords.length > 0 ? (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {existingKeywords.map((keyword, index) => (
                  <Chip
                    key={`saved-keyword-${index}-${keyword}`}
                    label={keyword}
                    color="error"
                    variant="filled"
                    size="small"
                    onDelete={() => removeKeyword(keyword)}
                    deleteIcon={<CloseIcon />}
                    sx={{
                      backgroundColor: '#d32f2f',
                      color: 'white',
                      '& .MuiChip-deleteIcon': {
                        color: 'white',
                        '&:hover': {
                          color: '#ffcdd2'
                        }
                      }
                    }}
                  />
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No saved keywords yet. Add keywords above and click "Save Keywords" to store them.
              </Typography>
            )}
            
            {/* Generate with Seed Keywords Button */}
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={generateContentIdeas}
                disabled={isGeneratingIdeas || !selectedTopicId || !user?.id || selectedSubtopics.length === 0 || existingKeywords.length === 0}
                size="large"
              >
                {isGeneratingIdeas ? 'Generating Ideas...' : 'Generate with Seed Keywords'}
              </Button>
            </Box>
          </Box>
        </Paper>

      {/* AHREFS Data Upload Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          AHREFS Data Upload
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Upload AHREFS CSV data to automatically generate content ideas with real keyword analytics
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <input
            accept=".csv,.tsv"
            style={{ display: 'none' }}
            id="ahrefs-file-upload"
            type="file"
            onChange={handleAhrefsFileUpload}
          />
          <label htmlFor="ahrefs-file-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<UploadIcon />}
              disabled={isUploadingAhrefs}
              sx={{ mr: 2 }}
            >
              {isUploadingAhrefs ? 'Uploading & Generating Ideas...' : 'Upload AHREFS CSV & Generate Ideas'}
            </Button>
          </label>
          
          {ahrefsFile && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Selected: {ahrefsFile.name}
            </Typography>
          )}
        </Box>

        {ahrefsKeywords.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              AHREFS Keywords ({ahrefsKeywords.length})
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              {ahrefsKeywords.slice(0, 10).map((keyword, index) => (
                <Chip
                  key={`ahrefs-keyword-${index}-${keyword.keyword}`}
                  label={`${keyword.keyword} (${keyword.volume})`}
                  color="primary"
                  variant="outlined"
                  size="small"
                />
              ))}
              {ahrefsKeywords.length > 10 && (
                <Chip
                  label={`+${ahrefsKeywords.length - 10} more`}
                  variant="outlined"
                  size="small"
                />
              )}
            </Box>
            
            {ahrefsAnalytics && (
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 2, mb: 2 }}>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    {ahrefsAnalytics.total_volume.toLocaleString()}
                  </Typography>
                  <Typography variant="caption">Total Volume</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    {ahrefsAnalytics.avg_difficulty}
                  </Typography>
                  <Typography variant="caption">Avg Difficulty</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    ${ahrefsAnalytics.avg_cpc}
                  </Typography>
                  <Typography variant="caption">Avg CPC</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    {ahrefsAnalytics.high_volume_keywords}
                  </Typography>
                  <Typography variant="caption">High Volume</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    {ahrefsAnalytics.low_difficulty_keywords}
                  </Typography>
                  <Typography variant="caption">Low Difficulty</Typography>
                </Box>
                <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="h6" color="primary">
                    {ahrefsAnalytics.commercial_keywords}
                  </Typography>
                  <Typography variant="caption">Commercial</Typography>
                </Box>
              </Box>
            )}
          </Box>
        )}
        
        {/* AHREFS Upload Status */}
        {ahrefsKeywords.length > 0 && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
            <Typography variant="body2" color="success.contrastText">
              ✅ AHREFS data uploaded successfully! Content ideas have been automatically generated using your keyword data.
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Advanced AHREFS Filters */}
      {showFilters && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Advanced Filters
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                onClick={() => setFilters({
                  minVolume: 1000,
                  maxVolume: 100000,
                  minDifficulty: 0,
                  maxDifficulty: 30,
                  minCpc: 0,
                  maxCpc: 100,
                  minTrafficPotential: 0,
                  maxTrafficPotential: 1000000,
                  minSeoScore: 70,
                  maxSeoScore: 100,
                  sortBy: 'score',
                  sortOrder: 'desc'
                })}
              >
                Easy Wins
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => setFilters({
                  minVolume: 10000,
                  maxVolume: 1000000,
                  minDifficulty: 0,
                  maxDifficulty: 50,
                  minCpc: 1,
                  maxCpc: 100,
                  minTrafficPotential: 0,
                  maxTrafficPotential: 1000000,
                  minSeoScore: 0,
                  maxSeoScore: 100,
                  sortBy: 'volume',
                  sortOrder: 'desc'
                })}
              >
                High Volume
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => setFilters({
                  minVolume: 0,
                  maxVolume: 1000000,
                  minDifficulty: 0,
                  maxDifficulty: 100,
                  minCpc: 5,
                  maxCpc: 100,
                  minTrafficPotential: 0,
                  maxTrafficPotential: 1000000,
                  minSeoScore: 0,
                  maxSeoScore: 100,
                  sortBy: 'cpc',
                  sortOrder: 'desc'
                })}
              >
                High CPC
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => setFilters({
                  minVolume: 0,
                  maxVolume: 1000000,
                  minDifficulty: 0,
                  maxDifficulty: 100,
                  minCpc: 0,
                  maxCpc: 100,
                  minTrafficPotential: 0,
                  maxTrafficPotential: 1000000,
                  minSeoScore: 0,
                  maxSeoScore: 100,
                  sortBy: 'score',
                  sortOrder: 'desc'
                })}
              >
                Reset
              </Button>
            </Box>
          </Box>
          
          <Grid container spacing={3}>
            {/* Search Volume Filter */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                Search Volume: {filters.minVolume.toLocaleString()} - {filters.maxVolume.toLocaleString()}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <input
                  type="range"
                  min="0"
                  max="1000000"
                  step="1000"
                  value={filters.minVolume}
                  onChange={(e) => {
                    const newMin = parseInt(e.target.value);
                    setFilters({...filters, minVolume: newMin, maxVolume: Math.max(newMin, filters.maxVolume)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #e0e0e0 0%, #e0e0e0 ${(filters.minVolume / 1000000) * 100}%, #1976d2 ${(filters.minVolume / 1000000) * 100}%, #1976d2 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max="1000000"
                  step="1000"
                  value={filters.maxVolume}
                  onChange={(e) => {
                    const newMax = parseInt(e.target.value);
                    setFilters({...filters, maxVolume: newMax, minVolume: Math.min(newMax, filters.minVolume)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #1976d2 0%, #1976d2 ${(filters.maxVolume / 1000000) * 100}%, #e0e0e0 ${(filters.maxVolume / 1000000) * 100}%, #e0e0e0 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
              </Box>
            </Grid>

            {/* Difficulty Filter */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                Difficulty: {filters.minDifficulty} - {filters.maxDifficulty}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.minDifficulty}
                  onChange={(e) => {
                    const newMin = parseInt(e.target.value);
                    setFilters({...filters, minDifficulty: newMin, maxDifficulty: Math.max(newMin, filters.maxDifficulty)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #e0e0e0 0%, #e0e0e0 ${(filters.minDifficulty / 100) * 100}%, #1976d2 ${(filters.minDifficulty / 100) * 100}%, #1976d2 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.maxDifficulty}
                  onChange={(e) => {
                    const newMax = parseInt(e.target.value);
                    setFilters({...filters, maxDifficulty: newMax, minDifficulty: Math.min(newMax, filters.minDifficulty)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #1976d2 0%, #1976d2 ${(filters.maxDifficulty / 100) * 100}%, #e0e0e0 ${(filters.maxDifficulty / 100) * 100}%, #e0e0e0 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
              </Box>
            </Grid>

            {/* CPC Filter */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                CPC: ${filters.minCpc.toFixed(2)} - ${filters.maxCpc.toFixed(2)}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="0.1"
                  value={filters.minCpc}
                  onChange={(e) => {
                    const newMin = parseFloat(e.target.value);
                    setFilters({...filters, minCpc: newMin, maxCpc: Math.max(newMin, filters.maxCpc)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #e0e0e0 0%, #e0e0e0 ${(filters.minCpc / 100) * 100}%, #1976d2 ${(filters.minCpc / 100) * 100}%, #1976d2 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="0.1"
                  value={filters.maxCpc}
                  onChange={(e) => {
                    const newMax = parseFloat(e.target.value);
                    setFilters({...filters, maxCpc: newMax, minCpc: Math.min(newMax, filters.minCpc)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #1976d2 0%, #1976d2 ${(filters.maxCpc / 100) * 100}%, #e0e0e0 ${(filters.maxCpc / 100) * 100}%, #e0e0e0 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
              </Box>
            </Grid>

            {/* Traffic Potential Filter */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                Traffic Potential: {filters.minTrafficPotential.toLocaleString()} - {filters.maxTrafficPotential.toLocaleString()}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <input
                  type="range"
                  min="0"
                  max="1000000"
                  step="1000"
                  value={filters.minTrafficPotential}
                  onChange={(e) => {
                    const newMin = parseInt(e.target.value);
                    setFilters({...filters, minTrafficPotential: newMin, maxTrafficPotential: Math.max(newMin, filters.maxTrafficPotential)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #e0e0e0 0%, #e0e0e0 ${(filters.minTrafficPotential / 1000000) * 100}%, #1976d2 ${(filters.minTrafficPotential / 1000000) * 100}%, #1976d2 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max="1000000"
                  step="1000"
                  value={filters.maxTrafficPotential}
                  onChange={(e) => {
                    const newMax = parseInt(e.target.value);
                    setFilters({...filters, maxTrafficPotential: newMax, minTrafficPotential: Math.min(newMax, filters.minTrafficPotential)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #1976d2 0%, #1976d2 ${(filters.maxTrafficPotential / 1000000) * 100}%, #e0e0e0 ${(filters.maxTrafficPotential / 1000000) * 100}%, #e0e0e0 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
              </Box>
            </Grid>

            {/* SEO Score Filter */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                SEO Score: {filters.minSeoScore} - {filters.maxSeoScore}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.minSeoScore}
                  onChange={(e) => {
                    const newMin = parseInt(e.target.value);
                    setFilters({...filters, minSeoScore: newMin, maxSeoScore: Math.max(newMin, filters.maxSeoScore)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #e0e0e0 0%, #e0e0e0 ${(filters.minSeoScore / 100) * 100}%, #1976d2 ${(filters.minSeoScore / 100) * 100}%, #1976d2 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.maxSeoScore}
                  onChange={(e) => {
                    const newMax = parseInt(e.target.value);
                    setFilters({...filters, maxSeoScore: newMax, minSeoScore: Math.min(newMax, filters.minSeoScore)});
                  }}
                  style={{ 
                    flex: 1,
                    background: `linear-gradient(to right, #1976d2 0%, #1976d2 ${(filters.maxSeoScore / 100) * 100}%, #e0e0e0 ${(filters.maxSeoScore / 100) * 100}%, #e0e0e0 100%)`,
                    WebkitAppearance: 'none',
                    appearance: 'none',
                    height: '6px',
                    borderRadius: '3px',
                    outline: 'none'
                  }}
                />
              </Box>
            </Grid>

            {/* Sort Options */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                Sort By
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {[
                  { value: 'score', label: 'Score' },
                  { value: 'volume', label: 'Volume' },
                  { value: 'difficulty', label: 'Difficulty' },
                  { value: 'cpc', label: 'CPC' },
                  { value: 'traffic_potential', label: 'Traffic' }
                ].map((option) => (
                  <Chip
                    key={option.value}
                    label={option.label}
                    color={filters.sortBy === option.value ? 'primary' : 'default'}
                    onClick={() => setFilters({...filters, sortBy: option.value as any})}
                    clickable
                    size="small"
                  />
                ))}
              </Box>
            </Grid>

            {/* Sort Order */}
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="subtitle2" gutterBottom>
                Order
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip
                  label="High to Low"
                  color={filters.sortOrder === 'desc' ? 'primary' : 'default'}
                  onClick={() => setFilters({...filters, sortOrder: 'desc'})}
                  clickable
                  size="small"
                />
                <Chip
                  label="Low to High"
                  color={filters.sortOrder === 'asc' ? 'primary' : 'default'}
                  onClick={() => setFilters({...filters, sortOrder: 'asc'})}
                  clickable
                  size="small"
                />
              </Box>
            </Grid>

            {/* Content Type Filter (Legacy) */}
            {session && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Content Type
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {['all', 'article', 'comparison', 'guide', 'tutorial'].map((type) => (
                    <Chip
                      key={type}
                      label={type}
                      color={session.filters.content_type === type ? 'primary' : 'default'}
                      onClick={() => setSession({
                        ...session,
                        filters: { ...session.filters, content_type: type }
                      })}
                      clickable
                      size="small"
                    />
                  ))}
                </Box>
              </Grid>
            )}
          </Grid>
        </Paper>
      )}

      {/* Filter Summary */}
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Showing {sortedIdeas.length} of {allIdeas.length} ideas
          {sortedIdeas.length !== allIdeas.length && ' (filtered)'}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<FilterIcon />}
            onClick={() => setShowFilters(!showFilters)}
            size="small"
          >
            {showFilters ? 'Hide' : 'Show'} Filters
          </Button>
        </Box>
      </Box>

      {/* Content Ideas Section */}
      {(contentIdeas.length > 0 || isGeneratingIdeas) && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Generated Content Ideas ({contentIdeas.length})
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant={selectedIdeaType === 'all' ? 'contained' : 'outlined'}
                onClick={() => setSelectedIdeaType('all')}
              >
                All ({contentIdeas.length})
              </Button>
              <Button
                size="small"
                variant={selectedIdeaType === 'blog' ? 'contained' : 'outlined'}
                onClick={() => setSelectedIdeaType('blog')}
              >
                Blog ({blogIdeas.length})
              </Button>
              <Button
                size="small"
                variant={selectedIdeaType === 'software' ? 'contained' : 'outlined'}
                onClick={() => setSelectedIdeaType('software')}
              >
                Software ({softwareIdeas.length})
              </Button>
            </Box>
          </Box>

          {isGeneratingIdeas && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Generating content ideas using AI...
              </Typography>
            </Box>
          )}

          {contentIdeas.length > 0 && (
            <Grid container spacing={2} key={refreshCounter}>
              {/* Select All Toggle for Publishable Ideas */}
              {(() => {
                const currentIdeas = selectedIdeaType === 'all' ? contentIdeas : 
                  selectedIdeaType === 'blog' ? blogIdeas : softwareIdeas;
                const publishableIdeas = currentIdeas.filter(idea => idea.content_type !== 'software');
                const selectedPublishableIdeas = publishableIdeas.filter(idea => selectedIdeasForPublish.has(idea.id));
                
                if (publishableIdeas.length === 0) return null;
                
                return (
                  <Grid item xs={12}>
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="h6">
                          Publish to Content Generation
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <Typography variant="body2" color="text.secondary">
                            {selectedPublishableIdeas.length} of {publishableIdeas.length} publishable ideas selected
                          </Typography>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => {
                              if (selectedPublishableIdeas.length === publishableIdeas.length) {
                                // Deselect all publishable ideas
                                const newSelected = new Set(selectedIdeasForPublish);
                                publishableIdeas.forEach(idea => newSelected.delete(idea.id));
                                setSelectedIdeasForPublish(newSelected);
                              } else {
                                // Select all publishable ideas
                                const newSelected = new Set(selectedIdeasForPublish);
                                publishableIdeas.forEach(idea => newSelected.add(idea.id));
                                setSelectedIdeasForPublish(newSelected);
                              }
                            }}
                          >
                            {selectedPublishableIdeas.length === publishableIdeas.length ? 'Deselect All' : 'Select All Publishable'}
                          </Button>
                          <Button
                            variant="contained"
                            startIcon={<PublishIcon />}
                            onClick={handlePublishToTitles}
                            disabled={selectedPublishableIdeas.length === 0 || isPublishing}
                            color="primary"
                          >
                            {isPublishing ? 'Publishing...' : 'Publish to Content Generation'}
                          </Button>
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                );
              })()}
              
        {(selectedIdeaType === 'all' ? contentIdeas : 
          selectedIdeaType === 'blog' ? blogIdeas : softwareIdeas).map((idea) => {
          const isSelectedForPublish = selectedIdeasForPublish.has(idea.id);
          const isPublishable = idea.content_type !== 'software' && !idea.published;
          const isPublished = Boolean(idea.published) || false;
          
          return (
                <Grid item xs={12} md={6} key={idea.id}>
                  <Card 
                    sx={{ 
                      height: '100%', 
                      display: 'flex', 
                      flexDirection: 'column',
                      border: isSelectedForPublish ? '2px solid' : '1px solid',
                      borderColor: isSelectedForPublish ? 'primary.main' : 'divider',
                      bgcolor: isSelectedForPublish ? 'primary.50' : 'background.paper',
                      position: 'relative'
                    }}
                  >
                    {/* Selected indicator */}
                    {isSelectedForPublish && (
                      <Box
                        sx={{
                          position: 'absolute',
                          bottom: 8,
                          right: 8,
                          bgcolor: 'primary.main',
                          color: 'white',
                          borderRadius: '50%',
                          width: 24,
                          height: 24,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '12px',
                          fontWeight: 'bold',
                          zIndex: 1
                        }}
                      >
                        ✓
                      </Box>
                    )}
                    
                    <CardContent sx={{ flexGrow: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                        <Box sx={{ flexGrow: 1, mr: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            {isPublishable ? (
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={isSelectedForPublish}
                                    onChange={(e) => handleIdeaCheckboxChange(idea.id, e.target.checked)}
                                    size="small"
                                  />
                                }
                                label={
                                  <Typography variant="h6" component="h3" sx={{ ml: 1 }}>
                                    {idea.title}
                                  </Typography>
                                }
                                sx={{ margin: 0, alignItems: 'flex-start' }}
                              />
                            ) : (
                              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6" component="h3" sx={{ mr: 1 }}>
                                  {idea.title}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        </Box>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={idea.content_type}
                      color={idea.content_type === 'blog' ? 'primary' : 'secondary'}
                      size="small"
                    />
                    {idea.generation_method && (
                      <Chip
                        label={idea.generation_method === 'llm' ? 'AI' : 'Template'}
                        color={idea.generation_method === 'llm' ? 'success' : 'default'}
                        size="small"
                        variant="outlined"
                      />
                    )}
                    {isPublished && (
                      <Chip
                        label="Published"
                        color="success"
                        size="small"
                        variant="filled"
                      />
                    )}
                  </Box>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {idea.description}
                      </Typography>

                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                        {(idea.keywords || []).slice(0, 3).map((keyword, index) => (
                          <Chip
                            key={`${idea.id}-keyword-${index}`}
                            label={keyword}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                        {(idea.keywords || []).length > 3 && (
                          <Chip
                            label={`+${(idea.keywords || []).length - 3} more`}
                            size="small"
                            variant="outlined"
                            color="default"
                          />
                        )}
                      </Box>

                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                        {idea.content_type === 'blog' ? (
                          <>
                            <Chip
                              label={`SEO: ${idea.seo_score || 0}`}
                              size="small"
                              color={idea.seo_score && idea.seo_score > 80 ? 'success' : 'default'}
                            />
                          </>
                        ) : (
                          <>
                            <Chip
                              label={idea.category}
                              size="small"
                              variant="outlined"
                            />
                          </>
                        )}
                        <Chip
                          label={idea.monetization_potential}
                          size="small"
                          color={idea.monetization_potential === 'high' ? 'success' : idea.monetization_potential === 'low' ? 'error' : 'default'}
                        />
                      </Box>

                      <Typography variant="caption" color="text.secondary">
                        Target: {idea.target_audience} • Subtopic: {idea.subtopic}
                      </Typography>
                    </CardContent>
                    
                    <CardActions sx={{ justifyContent: 'space-between' }}>
                      <Button
                        size="small"
                        color="error"
                        onClick={() => deleteContentIdea(idea.id)}
                      >
                        Delete
                      </Button>
                {isPublishable ? (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<PublishIcon />}
                    onClick={() => handlePublishSingleIdea(idea)}
                    disabled={isPublishing}
                    color="primary"
                  >
                    Publish
                  </Button>
                ) : null}
                    </CardActions>
                  </Card>
                </Grid>
                );
              })}
            </Grid>
          )}
        </Paper>
      )}


      {/* Ideas Grid */}
      {sortedIdeas.length > 0 ? (
        <Grid container spacing={3} key={refreshCounter}>
          {sortedIdeas.map((idea) => {
            const isSelected = selectedIdeas.has(idea.id);
            const isSelectedForPublish = selectedIdeasForPublish.has(idea.id);
            const combinedScore = (idea.seo_optimization_score + idea.traffic_potential_score) / 2;

            return (
              <Grid item xs={12} md={6} lg={4} key={idea.id}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  border: isSelectedForPublish ? '2px solid' : '1px solid',
                  borderColor: isSelectedForPublish ? 'primary.main' : 'divider',
                  bgcolor: isSelectedForPublish ? 'primary.50' : 'background.paper',
                  position: 'relative'
                }}
              >
                  {/* Selected indicator */}
                  {isSelectedForPublish && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        bgcolor: 'primary.main',
                        color: 'white',
                        borderRadius: '50%',
                        width: 24,
                        height: 24,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        fontWeight: 'bold',
                        zIndex: 1
                      }}
                    >
                      ✓
                    </Box>
                  )}
                  
                  <CardContent sx={{ flexGrow: 1 }}>
                    {/* Header */}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={isSelectedForPublish}
                                onChange={(e) => handleIdeaCheckboxChange(idea.id, e.target.checked)}
                                size="small"
                              />
                            }
                            label={
                              <Typography variant="h6" component="h3" sx={{ ml: 1 }}>
                                {idea.title}
                              </Typography>
                            }
                            sx={{ margin: 0, alignItems: 'flex-start' }}
                          />
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                          <Chip
                            label={`Score: ${combinedScore.toFixed(1)}`}
                            color={getScoreColor(combinedScore)}
                            size="small"
                          />
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Tooltip title="Add to favorites">
                          <IconButton size="small">
                            <StarIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Bookmark">
                          <IconButton size="small">
                            <BookmarkIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>

                    {/* Metrics */}
                    <Grid container spacing={2} sx={{ mb: 2 }}>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="success.main">
                            {formatNumber(idea.total_search_volume)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Volume
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="warning.main">
                            {idea.average_difficulty.toFixed(1)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Difficulty
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="info.main">
                            ${idea.average_cpc.toFixed(2)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            CPC
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Scores */}
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">SEO Score</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {idea.seo_optimization_score.toFixed(1)} ({getScoreLabel(idea.seo_optimization_score)})
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">Traffic Score</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {idea.traffic_potential_score.toFixed(1)} ({getScoreLabel(idea.traffic_potential_score)})
                        </Typography>
                      </Box>
                    </Box>

                    {/* Keywords */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Primary Keywords:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {idea.keywords.slice(0, 2).map((keyword) => (
                          <Chip
                            key={keyword}
                            label={keyword}
                            size="small"
                            color="primary"
                            icon={<SearchIcon />}
                          />
                        ))}
                      </Box>
                    </Box>

                    {/* Optimization Tips */}
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Key Tips:
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {idea.optimization_tips.slice(0, 2).join(' • ')}
                      </Typography>
                    </Box>
                  </CardContent>

                  <CardActions sx={{ justifyContent: 'space-between', p: 2 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => handleIdeaSelect(idea)}
                      color={isSelected ? 'primary' : 'inherit'}
                    >
                      {isSelected ? 'Selected' : 'Select'}
                    </Button>
                    <Box>
                      <Button
                        variant="contained"
                        size="small"
                        onClick={() => setShowIdeaDetails(idea.id)}
                        sx={{ mr: 1 }}
                      >
                        Details
                      </Button>
                      <IconButton size="small">
                        <ShareIcon />
                      </IconButton>
                    </Box>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      ) : !isGeneratingIdeas ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <LightbulbIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No ideas found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your filters or generate new ideas.
          </Typography>
          {/* Debug info */}
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Debug: sortedIdeas={sortedIdeas.length}, isGenerating={isGeneratingIdeas.toString()}
          </Typography>
        </Paper>
      ) : null}


      {/* Publish Result Dialog */}
      <Dialog
        open={publishDialogOpen}
        onClose={() => setPublishDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {isPublishing ? 'Publishing Ideas...' : 'Publish Results'}
        </DialogTitle>
        <DialogContent>
          {isPublishing ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <LinearProgress sx={{ mb: 2 }} />
              <Typography variant="body1">
                Publishing selected ideas to Content Generation table...
              </Typography>
            </Box>
          ) : publishResult ? (
            <Box>
              <Alert 
                severity={publishResult.success ? 'success' : 'error'} 
                sx={{ mb: 2 }}
              >
                {publishResult.success 
                  ? `Successfully published ${publishResult.published_count} ideas to Titles table`
                  : `Failed to publish ideas. ${publishResult.errors.length} errors occurred.`
                }
              </Alert>
              
              {publishResult.published_count > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Published Ideas:
                  </Typography>
                  <List dense>
                    {publishResult.published_titles.map((title, index) => (
                      <ListItem key={title.id}>
                        <ListItemText 
                          primary={`${index + 1}. ${title.Title}`}
                          secondary={`Keywords: ${title.Keywords}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
              
              {publishResult.errors.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom color="error">
                    Errors:
                  </Typography>
                  <List dense>
                    {publishResult.errors.map((error, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={error} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setPublishDialogOpen(false)}
            disabled={isPublishing}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={4000}
        onClose={() => setShowSnackbar(false)}
        message={snackbarMessage}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      />
    </Box>
  );
};

export default IdeaBurstPage;