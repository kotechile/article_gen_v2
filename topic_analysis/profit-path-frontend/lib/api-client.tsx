import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { supabase } from '@/lib/supabase';

// API Configuration
// API Configuration
const isClient = typeof window !== 'undefined';
// Client: use relative path for Nginx to handle. Server: use internal docker hostname.
// Client: ALWAYS use relative '/api' path to leverage Nginx proxy.
// Server: ALWAYS use internal Docker network URL 'http://backend:8000/api'.
// We ignore process.env.NEXT_PUBLIC_API_BASE_URL temporarily to prevent 0.0.0.0 issues.
const API_BASE_URL = isClient ? '/api' : 'http://backend:8000/api';
// In a real app, API_KEY should be in env, but for this demo/legacy pattern we might default it
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'dev-key';

class ApiClient {
    private client: AxiosInstance;

    constructor() {
        this.client = axios.create({
            baseURL: API_BASE_URL,
            timeout: 120000, // increased to 2 minutes
            headers: {
                'Content-Type': 'application/json',
            },
        });

        this.setupInterceptors();
    }

    private setupInterceptors() {
        // Request Interceptor
        this.client.interceptors.request.use(
            async (config) => {
                // 1. Add API Key (Backend Gatekeeper)
                if (!config.headers) {
                    config.headers = {} as any;
                }

                if (config.headers) {
                    config.headers['X-API-Key'] = API_KEY;
                }

                // 2. Add User ID if session exists (User Identity)
                try {
                    console.log("API Client: Retrieving session...");
                    // Timeout promise to prevent hanging - increased to 5000ms
                    const getSessionPromise = supabase.auth.getSession();
                    const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error('Session retrieval timed out after 5000ms')), 5000));

                    const result = await Promise.race([getSessionPromise, timeoutPromise]) as any;
                    const session = result.data?.session;

                    if (session?.access_token) {
                        console.log("API Client: Session found. Attaching token.");
                        if (config.headers) {
                            config.headers['Authorization'] = `Bearer ${session.access_token}`;
                        }
                    } else {
                        console.warn("API Client: No active session found.");
                    }
                } catch (err) {
                    console.error("API Client: Failed to get session for request", err);
                    // Decide if we should throw here to prevent unauthenticated requests
                    // For now, let's allow it but log clearly
                }

                console.log(`API Client: Config BaseURL: '${this.client.defaults.baseURL}'`);
                console.log(`API Client: Config URL: '${config.url}'`);
                // Calculate full URL for debugging
                const fullUrl = config.baseURL ? `${config.baseURL.replace(/\/$/, '')}/${config.url?.replace(/^\//, '')}` : config.url;
                console.log(`API Client: Sending ${config.method?.toUpperCase()} request to FULL URL: ${fullUrl}`);
                return config;
            },
            (error) => {
                console.error("API Client: Request interceptor error", error);
                return Promise.reject(error);
            }
        );

        // Response Interceptor
        this.client.interceptors.response.use(
            (response) => response,
            (error: AxiosError) => {
                if (error.response?.status === 401) {
                    // Handle unauthorized
                    console.warn("API Unauthorized");
                }
                return Promise.reject(error);
            }
        );
    }

    // Generic methods
    public async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
        const response = await this.client.get<T>(url, config);
        return response.data;
    }

    public async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
        const response = await this.client.post<T>(url, data, config);
        return response.data;
    }

    public async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
        const response = await this.client.put<T>(url, data, config);
        return response.data;
    }

    public async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
        const response = await this.client.delete<T>(url, config);
        return response.data;
    }
}

export const apiClient = new ApiClient();
