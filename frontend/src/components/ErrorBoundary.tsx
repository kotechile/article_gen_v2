import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, BookOpen } from 'lucide-react';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('[ErrorBoundary] Uncaught React render error:', error, errorInfo);
    }

    public handleReset = () => {
        this.setState({ hasError: false, error: null });
        window.location.reload();
    };

    public handleNavigateArticles = () => {
        this.setState({ hasError: false, error: null });
        window.location.href = '/my-articles';
    };

    public render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="min-h-[60vh] flex items-center justify-center p-6">
                    <div className="max-w-md w-full bg-card border border-border rounded-2xl p-6 shadow-xl text-center space-y-4">
                        <div className="w-12 h-12 rounded-full bg-destructive/10 text-destructive flex items-center justify-center mx-auto">
                            <AlertTriangle className="w-6 h-6" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-foreground">Something went wrong</h2>
                            <p className="text-sm text-muted-foreground mt-1">
                                An unexpected error occurred while rendering this page.
                            </p>
                        </div>
                        {this.state.error?.message && (
                            <div className="bg-muted/50 text-left p-3 rounded-xl border border-border text-xs font-mono text-muted-foreground max-h-28 overflow-auto">
                                {this.state.error.message}
                            </div>
                        )}
                        <div className="flex items-center justify-center gap-3 pt-2">
                            <button
                                type="button"
                                onClick={this.handleReset}
                                className="inline-flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground rounded-xl text-sm font-medium hover:bg-primary/90 transition shadow-sm"
                            >
                                <RefreshCw className="w-4 h-4" />
                                Reload
                            </button>
                            <button
                                type="button"
                                onClick={this.handleNavigateArticles}
                                className="inline-flex items-center gap-1.5 px-4 py-2 bg-muted hover:bg-muted/80 text-foreground border border-border rounded-xl text-sm font-medium transition"
                            >
                                <BookOpen className="w-4 h-4" />
                                Content Library
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
