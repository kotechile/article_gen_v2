# Google OAuth Setup for TrendTap

## 🎯 Your Supabase Configuration

Based on your Supabase project, here's what you need to configure:

### 1. Google Cloud Console Setup

1. **Go to [Google Cloud Console](https://console.cloud.google.com/)**
2. **Create a new project** or select existing one
3. **Enable Google+ API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Google+ API" and enable it
4. **Create OAuth 2.0 credentials**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add authorized redirect URIs:
     - `https://bvsqnmkvbbvtrcomtvnc.supabase.co/auth/v1/callback`
     - `http://localhost:3000/auth/callback` (for development)
   - Copy the **Client ID** and **Client Secret**

### 2. Supabase Dashboard Configuration

In your Supabase project dashboard:

1. **Go to Authentication → Providers**
2. **Enable Google provider**:
   - ✅ Toggle "Enable Sign in with Google"
   - **Client IDs**: Paste your Google Client ID
   - **Client Secret**: Paste your Google Client Secret
   - **Skip nonce checks**: Leave unchecked (more secure)
   - **Allow users without an email**: Leave unchecked
   - **Callback URL**: `https://bvsqnmkvbbvtrcomtvnc.supabase.co/auth/v1/callback` ✅

### 3. Frontend Environment Variables

Create a `.env` file in the frontend directory:

```bash
# Supabase Configuration
VITE_SUPABASE_URL=https://bvsqnmkvbbvtrcomtvnc.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ2c3FubWt2YmJ2dHJjb210dm5jIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk1MDYyMTQsImV4cCI6MjA3NTA4MjIxNH0.Vg6_r6djVh9vhwP6QNvg3HS5X4AI6Ic3EGp1BlHOeig

# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=TrendTap
VITE_DEBUG=true
```

## 🚀 Testing the Integration

1. **Start your application**:
   ```bash
   docker-compose -f docker-compose-local.yml up
   ```

2. **Visit the frontend**: `http://localhost:3000`

3. **You should see a login page with "Continue with Google" button**

4. **Click the button** and complete the Google OAuth flow

5. **After successful authentication**, you'll be redirected to the dashboard

## 🔧 Implementation Details

### Frontend Changes Made

1. **GoogleAuth Component**: Handles the OAuth flow
2. **AuthCallback Component**: Processes the OAuth callback
3. **App.withAuth.tsx**: Main app with authentication
4. **Supabase Client**: Configured for your project

### Backend Integration

The backend automatically works with Supabase Auth - no changes needed!

### User Flow

1. **User clicks "Continue with Google"**
2. **Redirected to Google OAuth**
3. **User authorizes the application**
4. **Google redirects back to Supabase**
5. **Supabase creates/updates user in database**
6. **Frontend receives user data and redirects to dashboard**

## 🔒 Security Features

- **Automatic user creation** in Supabase
- **JWT token management** handled by Supabase
- **Row Level Security (RLS)** automatically applies
- **Session management** integrated
- **User profile sync** from Google (name, email, avatar)

## 🛠️ Troubleshooting

### Common Issues:

1. **"Invalid redirect URI"**:
   - Check Google Cloud Console redirect URIs
   - Ensure they match exactly: `https://bvsqnmkvbbvtrcomtvnc.supabase.co/auth/v1/callback`

2. **"Client ID not found"**:
   - Verify Google Client ID in Supabase Dashboard
   - Check for typos or extra spaces

3. **"Access blocked"**:
   - Ensure Google+ API is enabled
   - Check OAuth consent screen is configured

### Debug Steps:

1. **Check Supabase logs**: Dashboard → Logs → Auth
2. **Check browser console** for JavaScript errors
3. **Verify environment variables** are loaded correctly

## 📱 User Experience

- **One-click Google sign-in**
- **Automatic account creation**
- **Profile picture and name** from Google
- **Seamless integration** with existing features
- **Secure session management**

---

**Ready to test! Let's switch to the authenticated version of the app.** 🚀


