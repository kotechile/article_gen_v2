/**
 * Enhanced Trends Analysis Page
 * Integrates with affiliate research to analyze trends for topics/subtopics
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  CircularProgress,
  Alert,
  LinearProgress,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  Edit,
  Save,
  Cancel,
  Add,
  Delete,
  Refresh,
  Analytics,
  CheckCircle,
  CloudUpload,
  AttachFile,
  Visibility,
  OpenInNew,
  FolderOpen,
} from '@mui/icons-material';
// import { useAffiliate } from '../hooks/useAffiliate';
// import { useTrends } from '../hooks/useTrends';

interface TrendData {
  id: string;
  topic: string;
  subtopic: string;
  trendScore: number;
  opportunityScore: number;
  searchVolume: number;
  competition: 'low' | 'medium' | 'high';
  trendDirection: 'up' | 'down' | 'stable';
  selected: boolean;
}

interface AffiliateResearch {
  id: string;
  main_topic: string;
  subtopics: string[];
  programs: any[];
  created_at: string;
}

interface TrendsAnalysisProps {
  currentResearch?: any;
  onTrendsSelected?: (trends: any[]) => void;
  onNavigateToTab?: (tabIndex: number) => void;
}

export const TrendsAnalysis: React.FC<TrendsAnalysisProps> = ({ currentResearch: propCurrentResearch, onTrendsSelected, onNavigateToTab }) => {
  // State management
  const [selectedResearchId, setSelectedResearchId] = useState<string>('');
  const [editableTopics, setEditableTopics] = useState<string[]>([]);
  const [editableSubtopics, setEditableSubtopics] = useState<string[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [trendsData, setTrendsData] = useState<TrendData[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedTrends, setSelectedTrends] = useState<string[]>([]);
  const [newTopic, setNewTopic] = useState('');
  const [newSubtopic, setNewSubtopic] = useState('');
  
  // File upload state
  const [uploadedFiles, setUploadedFiles] = useState<Record<string, File>>({});
  const [filePreview, setFilePreview] = useState<Record<string, any>>({});
  const [uploadedTrendsData, setUploadedTrendsData] = useState<Record<string, any>>({});
  const [multiFileUploads, setMultiFileUploads] = useState<Record<string, File[]>>({});
  const [googleTrendsLinks, setGoogleTrendsLinks] = useState<Record<string, string>>({});
  
  // Previous analyses state
  const [previousAnalyses, setPreviousAnalyses] = useState<any[]>([]);
  const [isLoadingPrevious, setIsLoadingPrevious] = useState(false);

  // Mock data for now - will be replaced with real API calls
  const researches: AffiliateResearch[] = [
    {
      id: '1',
      main_topic: 'running',
      subtopics: ['Running Shoes Reviews and Recommendations', 'Running Gear and Accessories', 'Nutrition for Runners', 'Running Training and Coaching Services', 'Running Apps and Technology', 'Running Events and Race Registrations'],
      programs: [],
      created_at: new Date().toISOString()
    },
    {
      id: '2',
      main_topic: 'telescope',
      subtopics: ['telescope accessories', 'astronomy equipment'],
      programs: [],
      created_at: new Date(Date.now() - 86400000).toISOString() // 1 day ago
    },
    {
      id: '3',
      main_topic: 'coffee',
      subtopics: ['coffee makers', 'espresso machines', 'coffee beans', 'coffee accessories'],
      programs: [],
      created_at: new Date(Date.now() - 172800000).toISOString() // 2 days ago
    },
    {
      id: '4',
      main_topic: 'fitness',
      subtopics: ['home gym equipment', 'fitness trackers', 'supplements', 'workout gear'],
      programs: [],
      created_at: new Date(Date.now() - 259200000).toISOString() // 3 days ago
    },
    {
      id: '5',
      main_topic: 'cooking',
      subtopics: ['kitchen appliances', 'cookware', 'cooking utensils', 'food processors'],
      programs: [],
      created_at: new Date(Date.now() - 345600000).toISOString() // 4 days ago
    }
  ];
  
  // Use passed research data or default to first research
  const currentResearch = propCurrentResearch || researches[0];

  // Auto-populate with current research or latest research
  useEffect(() => {
    console.log('📊 TrendsAnalysis received propCurrentResearch:', propCurrentResearch);
    if (propCurrentResearch) {
      // Use the passed research data
      console.log('📊 Using passed research data:', {
        main_topic: propCurrentResearch.main_topic,
        subtopics: propCurrentResearch.subtopics,
        programs: propCurrentResearch.programs?.length || 0
      });
      setSelectedResearchId('current');
      setEditableTopics([propCurrentResearch.main_topic]);
      setEditableSubtopics(propCurrentResearch.subtopics || []);
    } else if (researches.length > 0 && !selectedResearchId) {
      // Use the latest research from the list
      const latestResearch = researches[0];
      setSelectedResearchId(latestResearch.id);
      setEditableTopics([latestResearch.main_topic]);
      setEditableSubtopics(latestResearch.subtopics || []);
    }
  }, [propCurrentResearch, researches, selectedResearchId]);

  // Load research data into editable fields when selection changes
  useEffect(() => {
    if (selectedResearchId && selectedResearchId !== 'current') {
      const selectedResearch = researches.find(r => r.id === selectedResearchId);
      if (selectedResearch) {
        setEditableTopics([selectedResearch.main_topic]);
        setEditableSubtopics(selectedResearch.subtopics || []);
      }
    }
  }, [selectedResearchId, researches]);

  const handleResearchChange = (researchId: string) => {
    setSelectedResearchId(researchId);
    setTrendsData([]);
    setSelectedTrends([]);
    
    // Update topics and subtopics based on selection
    if (researchId === 'current' && propCurrentResearch) {
      setEditableTopics([propCurrentResearch.main_topic]);
      setEditableSubtopics(propCurrentResearch.subtopics || []);
    } else {
      const selectedResearch = researches.find(r => r.id === researchId);
      if (selectedResearch) {
        setEditableTopics([selectedResearch.main_topic]);
        setEditableSubtopics(selectedResearch.subtopics || []);
      }
    }
  };

  const handleEditToggle = () => {
    if (isEditing) {
      // Save changes
      setIsEditing(false);
    } else {
      setIsEditing(true);
    }
  };

  const handleTopicChange = (index: number, value: string) => {
    const newTopics = [...editableTopics];
    newTopics[index] = value;
    setEditableTopics(newTopics);
  };

  const handleSubtopicChange = (index: number, value: string) => {
    const newSubtopics = [...editableSubtopics];
    newSubtopics[index] = value;
    setEditableSubtopics(newSubtopics);
  };

  const handleAddTopic = () => {
    if (newTopic.trim()) {
      setEditableTopics([...editableTopics, newTopic.trim()]);
      setNewTopic('');
    }
  };

  const handleAddSubtopic = () => {
    if (newSubtopic.trim()) {
      setEditableSubtopics([...editableSubtopics, newSubtopic.trim()]);
      setNewSubtopic('');
    }
  };

  const handleDeleteTopic = (index: number) => {
    const newTopics = editableTopics.filter((_, i) => i !== index);
    setEditableTopics(newTopics);
  };

  const handleDeleteSubtopic = (index: number) => {
    const newSubtopics = editableSubtopics.filter((_, i) => i !== index);
    setEditableSubtopics(newSubtopics);
  };

  // File upload handlers
  const handleFileUpload = (subtopic: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && (file.type === 'text/csv' || file.type === 'application/json' || file.name.endsWith('.csv') || file.name.endsWith('.json'))) {
      setUploadedFiles(prev => ({ ...prev, [subtopic]: file }));
      
      // Parse file for preview and trends data
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        
        if (file.name.endsWith('.json')) {
          try {
            const jsonData = JSON.parse(content);
            setFilePreview(prev => ({ ...prev, [subtopic]: ['JSON Data:', JSON.stringify(jsonData, null, 2).split('\n').slice(0, 5)] }));
            setUploadedTrendsData(prev => ({ ...prev, [subtopic]: jsonData }));
          } catch (error) {
            setFilePreview(prev => ({ ...prev, [subtopic]: ['Invalid JSON file'] }));
          }
        } else {
          // CSV parsing
          const lines = content.split('\n').slice(0, 5);
          setFilePreview(prev => ({ ...prev, [subtopic]: lines }));
          
          // Try to parse as Google Trends CSV
          const parsedData = parseGoogleTrendsCSV(content);
          if (parsedData) {
            setUploadedTrendsData(prev => ({ ...prev, [subtopic]: parsedData }));
          }
        }
      };
      reader.readAsText(file);
    } else {
      alert('Please upload a CSV or JSON file');
    }
  };

  // Parse Google Trends CSV data
  const parseGoogleTrendsCSV = (csvContent: string) => {
    try {
      const lines = csvContent.split('\n').filter(line => line.trim());
      if (lines.length < 2) return null;

      const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      const data = lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
        const row: any = {};
        headers.forEach((header, index) => {
          row[header] = values[index] || '';
        });
        return row;
      });

      // Detect Google Trends data structure
      const isGoogleTrends = headers.some(h => 
        h.toLowerCase().includes('interest') || 
        h.toLowerCase().includes('region') || 
        h.toLowerCase().includes('topic') ||
        h.toLowerCase().includes('query')
      );

      if (isGoogleTrends) {
        return {
          type: 'google_trends',
          headers,
          data,
          summary: {
            totalRows: data.length,
            columns: headers,
            hasInterestData: headers.some(h => h.toLowerCase().includes('interest')),
            hasRegionData: headers.some(h => h.toLowerCase().includes('region')),
            hasTopicData: headers.some(h => h.toLowerCase().includes('topic'))
          }
        };
      }

      return {
        type: 'generic_csv',
        headers,
        data,
        summary: {
          totalRows: data.length,
          columns: headers
        }
      };
    } catch (error) {
      console.error('CSV parsing error:', error);
      return null;
    }
  };

  const handleRemoveFile = (subtopic: string) => {
    setUploadedFiles(prev => {
      const newFiles = { ...prev };
      delete newFiles[subtopic];
      return newFiles;
    });
    setFilePreview(prev => {
      const newPreview = { ...prev };
      delete newPreview[subtopic];
      return newPreview;
    });
  };

  const handleViewFile = (subtopic: string) => {
    const file = uploadedFiles[subtopic];
    if (file) {
      const url = URL.createObjectURL(file);
      window.open(url, '_blank');
    }
  };

  // Multi-file upload handlers
  const handleMultiFileUpload = (subtopic: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      setMultiFileUploads(prev => ({ ...prev, [subtopic]: files }));
      
      // Process each file
      files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          const fileType = detectGoogleTrendsFileType(file.name, content);
          
          setUploadedTrendsData(prev => ({
            ...prev,
            [subtopic]: {
              ...prev[subtopic],
              [fileType]: {
                filename: file.name,
                data: file.name.endsWith('.json') ? JSON.parse(content) : parseGoogleTrendsCSV(content),
                type: fileType
              }
            }
          }));
        };
        reader.readAsText(file);
      });
    }
  };

  // Detect Google Trends file type based on filename and content
  const detectGoogleTrendsFileType = (filename: string, content: string) => {
    const lowerFilename = filename.toLowerCase();
    
    if (lowerFilename.includes('interest_over_time') || lowerFilename.includes('trends')) {
      return 'interest_over_time';
    } else if (lowerFilename.includes('interest_by_region') || lowerFilename.includes('geographic')) {
      return 'interest_by_region';
    } else if (lowerFilename.includes('related_topics')) {
      return 'related_topics';
    } else if (lowerFilename.includes('related_queries')) {
      return 'related_queries';
    } else if (lowerFilename.includes('rising_queries')) {
      return 'rising_queries';
    } else {
      return 'general';
    }
  };

  // Google Trends link handlers
  const generateGoogleTrendsLink = (subtopic: string) => {
    const encodedQuery = encodeURIComponent(subtopic);
    return `https://trends.google.com/trends/explore?q=${encodedQuery}`;
  };

  const handleOpenGoogleTrends = (subtopic: string) => {
    const link = generateGoogleTrendsLink(subtopic);
    setGoogleTrendsLinks(prev => ({ ...prev, [subtopic]: link }));
    window.open(link, '_blank');
  };

  const handleAnalyzeTrends = async () => {
    if (editableTopics.length === 0 && editableSubtopics.length === 0) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setTrendsData([]);

    try {
      // Call the real backend API for trend analysis
      const response = await fetch('/api/workflow/trend-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topics: editableTopics,
          subtopics: editableSubtopics,
          user_id: 'demo-user'
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.trends) {
        // Transform the API response to match our TrendData interface
        const trends: TrendData[] = data.trends.map((trend: any, index: number) => ({
          id: trend.id || `trend-${index}`,
          topic: editableTopics.includes(trend.topic) ? trend.topic : '',
          subtopic: editableSubtopics.includes(trend.topic) ? trend.topic : '',
          trendScore: trend.trend_score || 0,
          opportunityScore: trend.opportunity_score || 0,
          searchVolume: trend.search_volume || 0,
          competition: trend.competition || 'medium',
          trendDirection: trend.trend_direction || 'stable',
          selected: false,
        }));
        
        setTrendsData(trends);
        console.log('✅ Real trend analysis completed:', trends.length, 'trends analyzed');
      } else {
        throw new Error(data.message || 'Failed to analyze trends');
      }
    } catch (error) {
      console.error('Failed to analyze trends:', error);
      // Show error message to user
      alert(`Trend analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}\n\nPlease check that the backend server is running and try again.`);
    } finally {
      setIsAnalyzing(false);
      setAnalysisProgress(100);
    }
  };

  const handleTrendSelection = (trendId: string) => {
    setSelectedTrends(prev => {
      const newSelection = prev.includes(trendId)
        ? prev.filter(id => id !== trendId)
        : [...prev, trendId];
      console.log('Trend selection updated:', { trendId, newSelection });
      
      // Update trendsData to reflect the selection
      const updatedTrendsData = trendsData.map(trend => 
        trend.id === trendId 
          ? { ...trend, selected: !trend.selected }
          : trend
      );
      setTrendsData(updatedTrendsData);
      
      // Get selected trend objects and pass to parent
      const selectedTrendObjects = updatedTrendsData.filter(trend => newSelection.includes(trend.id));
      if (onTrendsSelected) {
        onTrendsSelected(selectedTrendObjects);
      }
      
      return newSelection;
    });
  };

  const handleSelectAllProfitable = () => {
    const profitableTrends = trendsData
      .filter(trend => trend.opportunityScore >= 80)
      .map(trend => trend.id);
    setSelectedTrends(profitableTrends);
    
    // Keep trendsData.selected in sync with the bulk selection
    const updatedTrendsData = trendsData.map(trend => ({
      ...trend,
      selected: profitableTrends.includes(trend.id)
    }));
    setTrendsData(updatedTrendsData);
    
    // Get selected trend objects and pass to parent
    const selectedTrendObjects = updatedTrendsData.filter(trend => profitableTrends.includes(trend.id));
    if (onTrendsSelected) {
      onTrendsSelected(selectedTrendObjects);
    }
  };

  const handleGenerateContentIdeas = () => {
    // Get the selected trend objects
    const selectedTrendObjects = trendsData.filter(trend => selectedTrends.includes(trend.id));
    
    console.log('Selected trends for content generation:', selectedTrendObjects);
    console.log('Affiliate programs available:', currentResearch?.programs);
    
    // Pass the selected trends to the parent component
    if (onTrendsSelected) {
      onTrendsSelected(selectedTrendObjects);
    }
    
    // Navigate to the Idea Burst tab
    if (onNavigateToTab) {
      onNavigateToTab(3); // Tab 3 is Idea Burst
    }
  };

  // Load previous trend analyses
  const loadPreviousAnalyses = async () => {
    setIsLoadingPrevious(true);
    try {
      const response = await fetch(`/api/workflow/trend-analysis/demo-user`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.analyses) {
          setPreviousAnalyses(data.analyses);
          console.log('📊 Loaded previous analyses:', data.analyses.length);
        }
      } else {
        console.warn('⚠️ Failed to load previous analyses:', response.status);
      }
    } catch (error) {
      console.error('❌ Error loading previous analyses:', error);
    } finally {
      setIsLoadingPrevious(false);
    }
  };

  // Load previous analyses when component mounts
  useEffect(() => {
    loadPreviousAnalyses();
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getCompetitionColor = (competition: string) => {
    switch (competition) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };

  const getTrendDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'up': return '📈';
      case 'down': return '📉';
      default: return '➡️';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TrendingUp color="primary" />
        Trends Analysis
      </Typography>

      {/* Research Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Select Affiliate Research
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Previous Research</InputLabel>
            <Select
              value={selectedResearchId}
              onChange={(e) => handleResearchChange(e.target.value)}
              disabled={false}
            >
              {/* Show current research first if it exists */}
              {propCurrentResearch && (
                <MenuItem value="current">
                  {propCurrentResearch.main_topic} - Current Research
                </MenuItem>
              )}
              {/* Show other researches */}
              {researches.map((research) => (
                <MenuItem key={research.id} value={research.id}>
                  {research.main_topic} - {new Date(research.created_at).toLocaleDateString()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {currentResearch && (
            <Alert severity="info" sx={{ mb: 2 }}>
              Loaded research: <strong>{currentResearch.main_topic}</strong> with {currentResearch.subtopics?.length || 0} subtopics
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Topics and Subtopics Editor */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Topics & Subtopics
            </Typography>
            <Button
              variant={isEditing ? "contained" : "outlined"}
              startIcon={isEditing ? <Save /> : <Edit />}
              onClick={handleEditToggle}
            >
              {isEditing ? 'Save' : 'Edit'}
            </Button>
          </Box>

          {/* Topics Section */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Main Topics
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
              {editableTopics.map((topic, index) => (
                <Box
                  key={index}
                  sx={{
                    backgroundColor: '#1976d2',
                    color: 'white',
                    padding: '0.25rem 0.75rem',
                    borderRadius: '16px',
                    fontSize: '0.9rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}
                >
                  {topic}
                  {isEditing && (
                    <Button
                      onClick={() => handleDeleteTopic(index)}
                      sx={{
                        background: 'none',
                        border: 'none',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '1.2rem',
                        minWidth: 'auto',
                        padding: '0',
                        '&:hover': {
                          backgroundColor: 'rgba(255,255,255,0.2)'
                        }
                      }}
                    >
                      ×
                    </Button>
                  )}
                </Box>
              ))}
              
              {isEditing && (
                <Box sx={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <TextField
                    placeholder="New topic"
                    value={newTopic}
                    onChange={(e) => setNewTopic(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddTopic()}
                    size="small"
                    sx={{ width: '150px' }}
                  />
                  <Button
                    onClick={handleAddTopic}
                    sx={{
                      backgroundColor: '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '0.25rem 0.5rem',
                      cursor: 'pointer',
                      minWidth: 'auto',
                      '&:hover': {
                        backgroundColor: '#45a049'
                      }
                    }}
                  >
                    +
                  </Button>
                </Box>
              )}
            </Box>
          </Box>


          {/* Topics and Subtopics Section */}
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              Topics & Subtopics
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 2 }}>
              {editableSubtopics.map((subtopic, index) => (
                <Card key={index} sx={{ p: 2, backgroundColor: '#f8f9fa' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Box
                      sx={{
                        backgroundColor: '#9c27b0',
                        color: 'white',
                        padding: '0.25rem 0.75rem',
                        borderRadius: '16px',
                        fontSize: '0.9rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem'
                      }}
                    >
                      {subtopic}
                      {isEditing && (
                        <Button
                          onClick={() => handleDeleteSubtopic(index)}
                          sx={{
                            background: 'none',
                            border: 'none',
                            color: 'white',
                            cursor: 'pointer',
                            fontSize: '1.2rem',
                            minWidth: 'auto',
                            padding: '0',
                            '&:hover': {
                              backgroundColor: 'rgba(255,255,255,0.2)'
                            }
                          }}
                        >
                          ×
                        </Button>
                      )}
                    </Box>
                  </Box>
                  
                  {/* File Upload Section for each subtopic */}
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {/* Quick Actions Row */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                      {/* Google Trends Link Button */}
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<OpenInNew />}
                        onClick={() => handleOpenGoogleTrends(subtopic)}
                        sx={{ fontSize: '0.8rem', backgroundColor: '#1976d2' }}
                      >
                        Google Trends
                      </Button>
                      
                      {/* Single File Upload */}
                      <input
                        type="file"
                        accept=".csv,.json"
                        onChange={(e) => handleFileUpload(subtopic, e)}
                        style={{ display: 'none' }}
                        id={`file-upload-${index}`}
                      />
                      <label htmlFor={`file-upload-${index}`}>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<CloudUpload />}
                          component="span"
                          sx={{ fontSize: '0.8rem' }}
                        >
                          Upload Single File
                        </Button>
                      </label>
                      
                      {/* Multi-File Upload */}
                      <input
                        type="file"
                        accept=".csv,.json"
                        multiple
                        onChange={(e) => handleMultiFileUpload(subtopic, e)}
                        style={{ display: 'none' }}
                        id={`multi-file-upload-${index}`}
                      />
                      <label htmlFor={`multi-file-upload-${index}`}>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<FolderOpen />}
                          component="span"
                          sx={{ fontSize: '0.8rem' }}
                        >
                          Upload All Files
                        </Button>
                      </label>
                    </Box>
                    
                    {/* File Management */}
                    {uploadedFiles[subtopic] && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Visibility />}
                          onClick={() => handleViewFile(subtopic)}
                          sx={{ fontSize: '0.8rem' }}
                        >
                          View
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Delete />}
                          onClick={() => handleRemoveFile(subtopic)}
                          color="error"
                          sx={{ fontSize: '0.8rem' }}
                        >
                          Remove
                        </Button>
                        <Typography variant="caption" sx={{ color: 'green', fontWeight: 'bold' }}>
                          ✓ {uploadedFiles[subtopic].name}
                        </Typography>
                      </Box>
                    )}
                    
                    {/* Multi-File Status */}
                    {multiFileUploads[subtopic] && multiFileUploads[subtopic].length > 0 && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Typography variant="caption" sx={{ color: 'green', fontWeight: 'bold' }}>
                          ✓ {multiFileUploads[subtopic].length} files uploaded
                        </Typography>
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<Delete />}
                          onClick={() => {
                            setMultiFileUploads(prev => {
                              const newFiles = { ...prev };
                              delete newFiles[subtopic];
                              return newFiles;
                            });
                            setUploadedTrendsData(prev => {
                              const newData = { ...prev };
                              delete newData[subtopic];
                              return newData;
                            });
                          }}
                          color="error"
                          sx={{ fontSize: '0.8rem' }}
                        >
                          Clear All
                        </Button>
                      </Box>
                    )}
                  </Box>
                  
                  {/* File Preview & Trends Analysis */}
                  {filePreview[subtopic] && (
                    <Box sx={{ mt: 1, p: 1, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                        Data Preview:
                      </Typography>
                      <Box sx={{ fontFamily: 'monospace', fontSize: '0.75rem', mt: 0.5 }}>
                        {filePreview[subtopic].map((line: string, i: number) => (
                          <div key={i}>{line}</div>
                        ))}
                      </Box>
                    </Box>
                  )}

                  {/* Google Trends Data Analysis */}
                  {uploadedTrendsData[subtopic] && (
                    <Box sx={{ mt: 1, p: 1, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold', color: '#2e7d32' }}>
                        📊 Trends Analysis:
                      </Typography>
                      <Box sx={{ mt: 0.5 }}>
                        {/* Single file analysis */}
                        {uploadedTrendsData[subtopic].type === 'google_trends' && (
                          <>
                            <Typography variant="caption" sx={{ display: 'block', color: '#2e7d32' }}>
                              ✅ Google Trends Data Detected
                            </Typography>
                            <Typography variant="caption" sx={{ display: 'block' }}>
                              📈 Rows: {uploadedTrendsData[subtopic].summary.totalRows} | 
                              📊 Columns: {uploadedTrendsData[subtopic].summary.columns.length}
                            </Typography>
                            {uploadedTrendsData[subtopic].summary.hasInterestData && (
                              <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                                📈 Interest Over Time: Available
                              </Typography>
                            )}
                            {uploadedTrendsData[subtopic].summary.hasRegionData && (
                              <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                                🌍 Interest by Region: Available
                              </Typography>
                            )}
                            {uploadedTrendsData[subtopic].summary.hasTopicData && (
                              <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                                🔗 Related Topics: Available
                              </Typography>
                            )}
                          </>
                        )}
                        {uploadedTrendsData[subtopic].type === 'generic_csv' && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#666' }}>
                            📋 Generic CSV Data ({uploadedTrendsData[subtopic].summary.totalRows} rows)
                          </Typography>
                        )}
                        
                        {/* Multi-file analysis */}
                        {uploadedTrendsData[subtopic].interest_over_time && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                            📈 Interest Over Time: {uploadedTrendsData[subtopic].interest_over_time.filename}
                          </Typography>
                        )}
                        {uploadedTrendsData[subtopic].interest_by_region && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                            🌍 Interest by Region: {uploadedTrendsData[subtopic].interest_by_region.filename}
                          </Typography>
                        )}
                        {uploadedTrendsData[subtopic].related_topics && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                            🔗 Related Topics: {uploadedTrendsData[subtopic].related_topics.filename}
                          </Typography>
                        )}
                        {uploadedTrendsData[subtopic].related_queries && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                            🔍 Related Queries: {uploadedTrendsData[subtopic].related_queries.filename}
                          </Typography>
                        )}
                        {uploadedTrendsData[subtopic].rising_queries && (
                          <Typography variant="caption" sx={{ display: 'block', color: '#1976d2' }}>
                            📈 Rising Queries: {uploadedTrendsData[subtopic].rising_queries.filename}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  )}
                </Card>
              ))}
              
              {isEditing && (
                <Box sx={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <TextField
                    placeholder="New subtopic"
                    value={newSubtopic}
                    onChange={(e) => setNewSubtopic(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddSubtopic()}
                    size="small"
                    sx={{ width: '150px' }}
                  />
                  <Button
                    onClick={handleAddSubtopic}
                    sx={{
                      backgroundColor: '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '0.25rem 0.5rem',
                      cursor: 'pointer',
                      minWidth: 'auto',
                      '&:hover': {
                        backgroundColor: '#45a049'
                      }
                    }}
                  >
                    +
                  </Button>
                </Box>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Google Trends Data Guide */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 Google Trends Data Guide
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Upload Google Trends data to get rich insights including interest over time, geographic breakdown, and related topics.
          </Typography>
          
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 2 }}>
            <Box sx={{ p: 2, backgroundColor: '#f3e5f5', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#7b1fa2' }}>
                📈 Interest Over Time
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                Historical trend data showing search interest over time periods
              </Typography>
            </Box>
            
            <Box sx={{ p: 2, backgroundColor: '#e8f5e8', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#2e7d32' }}>
                🌍 Interest by Subregion
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                Geographic breakdown showing which regions have the most interest
              </Typography>
            </Box>
            
            <Box sx={{ p: 2, backgroundColor: '#fff3e0', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#f57c00' }}>
                🔗 Related Topics
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                Topics that are trending alongside your main topic
              </Typography>
            </Box>
            
            <Box sx={{ p: 2, backgroundColor: '#e3f2fd', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                🔍 Related Queries
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                Popular search queries related to your topic
              </Typography>
            </Box>
          </Box>
          
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>💡 Pro Tips:</strong>
            </Typography>
            <ul style={{ margin: '0.5rem 0 0 1rem', padding: 0 }}>
              <li>For broad topics like "running", use the main topic for trends analysis</li>
              <li>For specific subtopics like "running shoes", consider using broader terms or combining with main topic data</li>
              <li>Use the "Open Google Trends" button for quick access to live data</li>
              <li>Upload all Google Trends export files at once using "Upload All Files"</li>
              <li>Google Trends exports multiple files: Interest Over Time, Interest by Region, Related Topics, Related Queries</li>
            </ul>
          </Alert>
        </CardContent>
      </Card>

      {/* Analysis Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Trend Analysis
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                onClick={() => onNavigateToTab && onNavigateToTab(1)}
              >
                ← Back to Affiliate Research
              </Button>
              <Button
                variant="contained"
                startIcon={<Analytics />}
                onClick={handleAnalyzeTrends}
                disabled={isAnalyzing || (editableTopics.length === 0 && editableSubtopics.length === 0)}
              >
                Analyze Trends
              </Button>
              {trendsData.length > 0 && (
                <Button
                  variant="outlined"
                  startIcon={<CheckCircle />}
                  onClick={handleSelectAllProfitable}
                >
                  Select Profitable
                </Button>
              )}
            </Box>
          </Box>

          {isAnalyzing && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress variant="determinate" value={analysisProgress} />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Analyzing trends... {Math.round(analysisProgress)}%
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Trends Results */}
      {trendsData.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Trend Analysis Results
            </Typography>
            <Grid container spacing={2}>
              {trendsData.map((trend) => (
                <Grid item xs={12} md={6} lg={4} key={trend.id}>
                  <Paper
                    sx={{
                      p: 2,
                      border: selectedTrends.includes(trend.id) ? '2px solid' : '1px solid',
                      borderColor: selectedTrends.includes(trend.id) ? 'primary.main' : 'divider',
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                    onClick={() => handleTrendSelection(trend.id)}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {trend.topic || trend.subtopic}
                      </Typography>
                      <Checkbox
                        checked={selectedTrends.includes(trend.id)}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleTrendSelection(trend.id);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        size="small"
                      />
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                      <Chip
                        label={`Trend: ${trend.trendScore}%`}
                        color={getScoreColor(trend.trendScore)}
                        size="small"
                      />
                      <Chip
                        label={`Opportunity: ${trend.opportunityScore}%`}
                        color={getScoreColor(trend.opportunityScore)}
                        size="small"
                      />
                      <Chip
                        label={trend.competition}
                        color={getCompetitionColor(trend.competition)}
                        size="small"
                      />
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        Volume: {trend.searchVolume.toLocaleString()}
                      </Typography>
                      <Typography variant="body2">
                        {getTrendDirectionIcon(trend.trendDirection)} {trend.trendDirection}
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>

            {selectedTrends.length > 0 && (
              <Box sx={{ mt: 3, p: 2, backgroundColor: 'primary.light', borderRadius: 1 }}>
                <Typography variant="h6" gutterBottom>
                  Selected Trends ({selectedTrends.length})
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  These trends will be used for content idea generation along with your affiliate programs.
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => {
                    // Navigate to Idea Burst tab (tab 3)
                    if (onNavigateToTab) {
                      onNavigateToTab(3);
                    }
                  }}
                  sx={{ 
                    mt: 1,
                    backgroundColor: '#2e7d32',
                    '&:hover': {
                      backgroundColor: '#1b5e20'
                    }
                  }}
                >
                  💡 Start Idea Generation
                </Button>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Loading States - Removed for now */}
    </Box>
  );
};

export default TrendsAnalysis;
