import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { Loader2 } from 'lucide-react';

export const AuthCallback: React.FC = () => {
    const navigate = useNavigate();

    useEffect(() => {
        const handleAuthCallback = async () => {
            console.log('AuthCallback: Processing login redirect...');
            const hash = window.location.hash;
            console.log('AuthCallback: Hash present:', !!hash);

            if (!hash) {
                console.warn('AuthCallback: No hash found.');
                // Allow a small grace period in case the client strips it incredibly fast, 
                // but typically if it's gone, it's gone.
                // Try getSession one last time.
                const { data: { session } } = await supabase.auth.getSession();
                if (session) {
                    navigate('/');
                    return;
                }
                navigate('/login?error=no_hash');
                return;
            }

            try {
                // Manual Hash Parsing
                const params = new URLSearchParams(hash.substring(1)); // remove #
                const accessToken = params.get('access_token');
                const refreshToken = params.get('refresh_token');

                if (accessToken && refreshToken) {
                    console.log('AuthCallback: Found tokens in hash. Manually setting session...');
                    const { data, error } = await supabase.auth.setSession({
                        access_token: accessToken,
                        refresh_token: refreshToken,
                    });

                    if (error) {
                        console.error('AuthCallback: Error setting session:', error);
                        navigate(`/login?error=${encodeURIComponent(error.message)}`);
                        return;
                    }

                    if (data.session) {
                        console.log('AuthCallback: Session set successfully! Redirecting...');
                        navigate('/');
                        return;
                    }
                } else {
                    console.log('AuthCallback: Access/Refresh token missing from hash.');
                }

                // Fallback to auto-detection if manual parsing failed or wasn't needed
                const { data: { session }, error } = await supabase.auth.getSession();

                if (error) throw error;

                if (session) {
                    console.log('AuthCallback: Session found via getSession! Redirecting.');
                    navigate('/');
                } else {
                    console.warn('AuthCallback: No session found after processing.');
                    navigate('/login?error=no_session_set');
                }
            } catch (err: any) {
                console.error('AuthCallback: Unexpected error', err);
                navigate(`/login?error=${encodeURIComponent(err.message || 'Unknown error')}`);
            }
        };

        handleAuthCallback();
    }, [navigate]);

    return (
        <div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-600 mb-4" />
            <p className="text-gray-500 font-medium">Completing secure sign in...</p>
        </div>
    );
};
