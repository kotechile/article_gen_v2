/**
 * Idea Burst Page Component
 * Content and software idea generation workflow
 */

import React from 'react';
import { useLocation } from 'react-router-dom';
import { Box } from '@mui/material';
import IdeaBurstPage from '../components/IdeaBurstPage';

export const IdeaBurst: React.FC = () => {
  const location = useLocation();
  const navigationState = location.state as { 
    selectedTopicId?: string; 
    selectedTopicTitle?: string; 
    selectedSubtopics?: string[] 
  } | null;

  return (
    <Box sx={{ flexGrow: 1 }}>
      <IdeaBurstPage 
        selectedTopicId={navigationState?.selectedTopicId}
        selectedTopicTitle={navigationState?.selectedTopicTitle}
        selectedSubtopics={navigationState?.selectedSubtopics}
      />
    </Box>
  );
};
