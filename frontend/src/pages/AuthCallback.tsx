import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { Loader2 } from 'lucide-react';

export const AuthCallback: React.FC = () => {
    const navigate = useNavigate();

    useEffect(() => {
        const handleAuthCallback = async () => {
            console.log('AuthCallback: Processing login redirect...');
            console.log('AuthCallback: Hash:', window.location.hash);

            try {
                // With implicit flow, the session is in the URL.
                // supabase.auth.getSession() should pick it up automatically.
                // We just need to wait a moment or verify it.

                const { data: { session }, error } = await supabase.auth.getSession();

                if (error) {
                    console.error('AuthCallback: Error getting session', error);
                    navigate('/login?error=auth_callback_failed');
                    return;
                }

                if (session) {
                    console.log('AuthCallback: Session found! Redirecting to dashboard.');
                    navigate('/');
                } else {
                    console.warn('AuthCallback: No session found after redirect. Waiting for onAuthStateChange...');
                    // If getSession didn't pick it up, onAuthStateChange might.
                    // But if we return here, we might get stuck. 
                    // Let's rely on AuthProvider's global listener to handle the 'SIGNED_IN' event 
                    // and redirect us. Or we can force a check.

                    // Fallback: redirects to login if nothing happens after a delay
                    setTimeout(() => {
                        navigate('/login?error=no_session_found');
                    }, 3000);
                }
            } catch (err) {
                console.error('AuthCallback: Unexpected error', err);
                navigate('/login');
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
