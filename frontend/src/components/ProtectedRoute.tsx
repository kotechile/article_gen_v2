
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/auth-context';
import { Loader2 } from 'lucide-react';

export const ProtectedRoute: React.FC = () => {
    const { user, isLoading } = useAuth();

    console.log(`ProtectedRoute: isLoading=${isLoading}, user=${user ? 'FOUND' : 'NULL'}`);

    if (isLoading) {
        return (
            <div className="h-screen w-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return <Outlet />;
};
