"""
Streamlit Authentication Integration

This module provides Streamlit-specific authentication components including:
- Login/logout UI components
- Session management
- OAuth callback handling
- User dashboard
- Rate limiting displays

Author: Claude Code
"""

import streamlit as st
import logging
from typing import Optional
from auth_system import auth_system, UserProfile
from urllib.parse import urlparse, parse_qs
import time

logger = logging.getLogger(__name__)

class StreamlitAuthManager:
    """Streamlit-specific authentication manager"""
    
    def __init__(self):
        self.auth_system = auth_system
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state for authentication"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = None
        if 'auth_provider' not in st.session_state:
            st.session_state.auth_provider = None
    
    def render_login_page(self):
        """Render the login page with OAuth options"""
        st.title("🔐 CPUC RAG System - Login")
        
        st.markdown("""
        ### Welcome to the CPUC Regulatory Document Analysis System
        
        Please sign in with your organization account to access the system.
        Different organizations have different access levels:
        
        - **Partner Organizations** (e.g., @sanjosecleanenergy.org): 3 queries per day
        - **Regulatory Bodies** (e.g., @cpuc.ca.gov): 10 queries per day  
        - **General Users**: 1 query per day
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 Microsoft Teams")
            if st.button("Sign in with Microsoft", key="microsoft_login", type="primary"):
                auth_url = self.auth_system.get_microsoft_auth_url()
                st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
                st.info("Redirecting to Microsoft login...")
        
        with col2:
            st.subheader("🔍 Google")
            if st.button("Sign in with Google", key="google_login", type="primary"):
                auth_url = self.auth_system.get_google_auth_url()
                st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
                st.info("Redirecting to Google login...")
        
        # Handle OAuth callbacks
        self.handle_oauth_callback()
    
    def handle_oauth_callback(self):
        """Handle OAuth callback from URL parameters"""
        # Get URL parameters
        try:
            query_params = st.query_params
            
            # Microsoft callback
            if 'code' in query_params and 'state' not in query_params:
                st.info("Processing Microsoft authentication...")
                auth_code = query_params['code'][0]
                user_profile = self.auth_system.handle_microsoft_callback(auth_code)
                
                if user_profile:
                    self.login_user(user_profile, 'microsoft')
                    st.query_params.clear()  # Clear URL parameters
                    st.rerun()
                else:
                    st.error("Microsoft authentication failed. Please try again.")
            
            # Google callback
            elif 'code' in query_params and 'scope' in query_params:
                st.info("Processing Google authentication...")
                auth_code = query_params['code'][0]
                user_profile = self.auth_system.handle_google_callback(auth_code)
                
                if user_profile:
                    self.login_user(user_profile, 'google')
                    st.query_params.clear()  # Clear URL parameters
                    st.rerun()
                else:
                    st.error("Google authentication failed. Please try again.")
        
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
    
    def login_user(self, user_profile: UserProfile, provider: str):
        """Log in a user and set session state"""
        st.session_state.authenticated = True
        st.session_state.user_profile = user_profile
        st.session_state.auth_provider = provider
        st.success(f"Welcome, {user_profile.name}!")
        
        # Log the login
        logger.info(f"User logged in: {user_profile.email} via {provider}")
    
    def logout_user(self):
        """Log out the current user"""
        if st.session_state.user_profile:
            logger.info(f"User logged out: {st.session_state.user_profile.email}")
        
        st.session_state.authenticated = False
        st.session_state.user_profile = None
        st.session_state.auth_provider = None
        st.rerun()
    
    def render_user_dashboard(self):
        """Render user dashboard with account info and usage stats"""
        if not st.session_state.authenticated:
            return
        
        user_profile = st.session_state.user_profile
        user_stats = self.auth_system.get_user_stats(user_profile.email)
        
        # Header with user info and logout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"👋 Welcome, {user_profile.name}")
        with col2:
            if st.button("🚪 Logout", key="logout_btn"):
                self.logout_user()
        
        # User stats dashboard
        st.markdown("### 📊 Your Account")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Access Tier",
                value=user_profile.access_tier.title(),
                help=f"Your access level: {user_stats['access_tier_info']['description']}"
            )
        
        with col2:
            st.metric(
                label="Daily Limit",
                value=user_profile.daily_query_limit,
                help="Maximum queries allowed per day"
            )
        
        with col3:
            remaining = user_stats['remaining_queries']
            st.metric(
                label="Remaining Today",
                value=remaining,
                delta=f"-{user_profile.queries_used_today} used",
                help="Queries remaining for today"
            )
        
        with col4:
            st.metric(
                label="Total Queries",
                value=user_stats['total_queries'],
                help="All-time query count"
            )
        
        # Usage warning
        if remaining <= 0:
            st.error("⚠️ You have reached your daily query limit. Please try again tomorrow.")
        elif remaining <= 1:
            st.warning("⚠️ You have 1 query remaining today. Use it wisely!")
        
        st.markdown("---")
    
    def check_query_permission(self) -> tuple[bool, str]:
        """Check if user can make a query"""
        if not st.session_state.authenticated:
            return False, "Please log in to use the system"
        
        user_profile = st.session_state.user_profile
        can_query, message = self.auth_system.can_make_query(user_profile.email)
        
        return can_query, message
    
    def log_query(self, query_text: str, confidence_score: int = None, sources_count: int = None):
        """Log a query for the current user"""
        if not st.session_state.authenticated:
            return False
        
        user_profile = st.session_state.user_profile
        success = self.auth_system.log_query(
            user_profile.email, 
            query_text, 
            confidence_score, 
            sources_count
        )
        
        if success:
            # Update session state
            user_profile.queries_used_today += 1
            st.session_state.user_profile = user_profile
        
        return success
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.authenticated
    
    def get_current_user(self) -> Optional[UserProfile]:
        """Get current user profile"""
        return st.session_state.user_profile if st.session_state.authenticated else None

# Global Streamlit auth manager
streamlit_auth = StreamlitAuthManager()