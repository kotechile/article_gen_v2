import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/auth-context';
import { ProjectProvider } from './context/project-context';
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
import { TopicDetail } from './pages/TopicDetail';
import { Settings } from './pages/Settings';

function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/auth/callback" element={<AuthCallback />} />

        <Route element={<ProtectedRoute />}>
          {/* ProjectProvider is inside ProtectedRoute so user is guaranteed */}
          <Route element={<ProjectProvider><Landing /></ProjectProvider>} path="/" />

          {/* Main Layout routes — also wrapped with ProjectProvider */}
          <Route element={<ProjectProvider><MainLayout /></ProjectProvider>}>
            <Route path="/my-articles" element={<MyArticles />} />
            <Route path="/knowledge-gaps" element={<KnowledgeGaps />} />
            <Route path="/content-studio" element={<ContentStudio />} />
            <Route path="/article-editor/:id" element={<ArticleEditor />} />
            <Route path="/research" element={<Research />} />
            <Route path="/research/:id" element={<TopicDetail />} />
            <Route path="/history" element={<div>History (Coming Soon)</div>} />
            <Route path="/settings" element={<Settings />} />
          </Route>
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  );
}

export default App;
