# Google OAuth Integration with Supabase

This guide will help you set up Google OAuth authentication for TrendTap using Supabase Auth.

## 🚀 Quick Setup (5 minutes)

### Step 1: Google Cloud Console Setup

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
     - `https://your-project-ref.supabase.co/auth/v1/callback`
     - `http://localhost:3000/auth/callback` (for development)
   - Copy the **Client ID** and **Client Secret**

### Step 2: Supabase Dashboard Setup

1. **Go to your Supabase project dashboard**
2. **Navigate to Authentication → Providers**
3. **Enable Google provider**:
   - Toggle "Enable Google provider"
   - Paste your Google **Client ID**
   - Paste your Google **Client Secret**
   - Set redirect URL to: `https://your-project-ref.supabase.co/auth/v1/callback`
4. **Save the configuration**

### Step 3: Environment Variables

Add these to your `.env` file:

```bash
# Google OAuth (already configured in Supabase)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Supabase Auth URLs
SUPABASE_AUTH_URL=https://your-project-ref.supabase.co/auth/v1
SUPABASE_REDIRECT_URL=http://localhost:3000/auth/callback
```

## 🔧 Implementation Details

### Backend Changes

The backend will automatically handle Google OAuth tokens through Supabase Auth. No additional backend changes needed!

### Frontend Changes

The frontend will use Supabase Auth client to handle Google OAuth flow.

## 🧪 Testing

1. **Start your application**:
   ```bash
   docker-compose -f docker-compose-local.yml up
   ```

2. **Visit the frontend**: `http://localhost:3000`

3. **Click "Sign in with Google"** button

4. **Complete Google OAuth flow**

5. **Verify user is created** in Supabase Dashboard → Authentication → Users

## 🔒 Security Features

- **Automatic user creation** in Supabase
- **JWT token management** handled by Supabase
- **Row Level Security (RLS)** automatically applies
- **Session management** integrated
- **User profile sync** from Google

## 📱 User Experience

- **One-click Google sign-in**
- **Automatic account creation**
- **Profile picture and name** from Google
- **Seamless integration** with existing features

## 🛠️ Troubleshooting

### Common Issues:

1. **"Invalid redirect URI"**:
   - Check Google Cloud Console redirect URIs
   - Ensure they match exactly (including trailing slashes)

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

## 🎯 Next Steps

After setup:
1. **Test the Google OAuth flow**
2. **Verify user data sync**
3. **Test RLS policies** with Google-authenticated users
4. **Customize user profile** handling if needed

---

**Ready to implement? Let's start with the frontend components!** 🚀


