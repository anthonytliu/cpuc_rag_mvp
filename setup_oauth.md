# OAuth Setup Guide

This guide will help you set up OAuth authentication for Microsoft Teams and Google.

## Microsoft Teams / Azure AD Setup

### 1. Create Azure AD Application
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to "Azure Active Directory" > "App registrations"
3. Click "New registration"
4. Fill in:
   - **Name**: CPUC RAG System
   - **Supported account types**: Accounts in any organizational directory (Any Azure AD directory - Multitenant)
   - **Redirect URI**: Web - `http://localhost:8501/auth/microsoft/callback`

### 2. Configure Application
1. Go to "API permissions"
   - Add Microsoft Graph permissions:
     - `openid`
     - `profile`
     - `email`
     - `User.Read`
2. Go to "Certificates & secrets"
   - Create a new client secret
   - Copy the secret value (you won't see it again)

### 3. Environment Variables
```bash
MICROSOFT_CLIENT_ID=your_application_id_here
MICROSOFT_CLIENT_SECRET=your_client_secret_here
MICROSOFT_REDIRECT_URI=http://localhost:8501/auth/microsoft/callback
```

## Google OAuth Setup

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google+ API and Google Identity API

### 2. Create OAuth Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Web application"
4. Fill in:
   - **Name**: CPUC RAG System
   - **Authorized redirect URIs**: `http://localhost:8501/auth/google/callback`

### 3. Environment Variables
```bash
GOOGLE_CLIENT_ID=your_client_id_here
GOOGLE_CLIENT_SECRET=your_client_secret_here
GOOGLE_REDIRECT_URI=http://localhost:8501/auth/google/callback
```

## Local Development Setup

### 1. Environment File
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your OAuth credentials
nano .env
```

### 2. Install Dependencies
```bash
pip install -r requirements_auth.txt
```

### 3. Test Authentication
```bash
# Run the app
streamlit run app.py

# Navigate to http://localhost:8501
# Try logging in with both Microsoft and Google
```

## Production Deployment

### 1. Update Redirect URIs
- Microsoft: `https://your-domain.com/auth/microsoft/callback`
- Google: `https://your-domain.com/auth/google/callback`

### 2. Environment Variables
```bash
MICROSOFT_REDIRECT_URI=https://your-domain.com/auth/microsoft/callback
GOOGLE_REDIRECT_URI=https://your-domain.com/auth/google/callback
DOMAIN=your-domain.com
```

### 3. Security Considerations
- Use HTTPS in production
- Set strong secret keys
- Use environment variables (not hardcoded)
- Consider using a proper database (PostgreSQL) instead of SQLite
- Implement proper session management
- Add rate limiting and DDoS protection

## Access Tier Configuration

The system automatically assigns access tiers based on email domains:

### Free Tier (1 query/day)
- gmail.com
- outlook.com  
- yahoo.com

### Partner Tier (3 queries/day)
- sanjosecleanenergy.org
- cpuc.ca.gov
- energy.ca.gov

### Premium Tier (10 queries/day)
- cpuc.ca.gov
- energy.ca.gov
- gov.ca.gov

### Unlimited Tier (999999 queries/day)
- admin.cpuc.ca.gov
- dev.cpuc.ca.gov

To modify these tiers, edit the `load_access_tiers()` method in `auth_system.py`.

## Troubleshooting

### Common Issues

1. **OAuth callback not working**
   - Check redirect URI matches exactly
   - Ensure proper URL encoding
   - Check firewall/network settings

2. **Authentication fails**
   - Verify client ID and secret
   - Check API permissions
   - Look at browser console for errors

3. **Database errors**
   - Ensure SQLite file permissions
   - Check database file path
   - Verify database initialization

4. **Rate limiting not working**
   - Check system clock/timezone
   - Verify database updates
   - Check user session state

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
streamlit run app.py --logger.level=debug
```

## Security Best Practices

1. **Never commit secrets to version control**
2. **Use environment variables for all sensitive data**
3. **Implement proper session timeout**
4. **Use HTTPS in production**
5. **Regularly rotate OAuth secrets**
6. **Monitor authentication logs**
7. **Implement proper error handling**
8. **Use rate limiting and DDoS protection**