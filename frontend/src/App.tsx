import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider } from './context/auth-context';
import { ProjectProvider } from './context/project-context';
import { ThemeProvider } from './components/theme-provider';
import { AuthCallback } from './pages/AuthCallback';
import { Login } from './pages/Login';
import { Landing } from './pages/Landing';
import { ProtectedRoute } from './components/ProtectedRoute';
import MainLayout from './components/layout/MainLayout';
import { MyArticles } from './pages/MyArticles';
import { ContentStudio } from './pages/ContentStudio';
import { KnowledgeGaps } from './pages/KnowledgeGaps';
import { ArticleEditor } from './pages/ArticleEditor';
import { Research } from './pages/Research';
import { ResearchRebuildStrategicPage } from './pages/ResearchRebuildStrategic';
import { TopicDetail } from './pages/TopicDetail';
import { Settings } from './pages/Settings';
import { SoftwareIdeas } from './pages/SoftwareIdeas';

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/auth/callback" element={<AuthCallback />} />

            <Route element={<ProtectedRoute />}>
            <Route element={<ProjectProvider><MainLayout /></ProjectProvider>}>
              <Route path="/" element={<ResearchRebuildStrategicPage />} />
              <Route path="/research-rebuild" element={<ResearchRebuildRedirect />} />
              <Route path="/research-rebuild/jobs" element={<ResearchRebuildStrategicPage />} />
              <Route path="/research-rebuild/opportunities" element={<ResearchRebuildStrategicPage />} />
              <Route path="/my-articles" element={<MyArticles />} />
              <Route path="/software-ideas" element={<SoftwareIdeas />} />
              <Route path="/knowledge-gaps" element={<KnowledgeGaps />} />
              <Route path="/content-studio" element={<ContentStudio />} />
              <Route path="/article-editor/:id" element={<ArticleEditor />} />
              <Route path="/legacy-research" element={<Research />} />
              <Route path="/legacy-new-research" element={<Landing />} />
              <Route path="/research/:id" element={<TopicDetail />} />
              <Route path="/history" element={<div>History (Coming Soon)</div>} />
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </ThemeProvider>
  );
}

function ResearchRebuildRedirect() {
  const location = useLocation();
  const target = '/research-rebuild/jobs';
  return <Navigate to={`${target}${location.search || ''}`} replace />;
}

export default App;
