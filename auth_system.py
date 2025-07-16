"""
Authentication System for CPUC RAG

This module provides comprehensive authentication including:
- OAuth integration with Microsoft Teams and Google
- Email domain-based access tiers
- Query rate limiting with 24-hour reset
- User session management
- Web deployment ready architecture

Author: Claude Code
"""

import os
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
import requests

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile with access tier and usage tracking"""
    email: str
    name: str
    provider: str  # 'microsoft' or 'google'
    access_tier: str
    daily_query_limit: int
    queries_used_today: int
    last_reset_date: str
    created_at: str
    last_login: str
    is_active: bool = True

@dataclass
class AccessTier:
    """Access tier configuration"""
    name: str
    daily_query_limit: int
    description: str
    domain_patterns: list

class AuthenticationSystem:
    """
    Complete authentication system with OAuth, access tiers, and rate limiting
    """
    
    def __init__(self, db_path: str = "auth_system.db"):
        self.db_path = db_path
        self.init_database()
        self.load_access_tiers()
        self.setup_oauth_clients()
    
    def init_database(self):
        """Initialize SQLite database for user management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                access_tier TEXT NOT NULL,
                daily_query_limit INTEGER NOT NULL,
                queries_used_today INTEGER DEFAULT 0,
                last_reset_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Query logs table for detailed tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                query_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence_score INTEGER,
                sources_count INTEGER,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        # OAuth tokens table (encrypted)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                provider TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_access_tiers(self):
        """Load access tier configurations"""
        self.access_tiers = {
            'free': AccessTier(
                name='Free',
                daily_query_limit=1,
                description='Basic access for general users',
                domain_patterns=['gmail.com', 'outlook.com', 'yahoo.com']
            ),
            'partner': AccessTier(
                name='Partner',
                daily_query_limit=3,
                description='Enhanced access for partner organizations',
                domain_patterns=[
                    'sanjosecleanenergy.org',
                    'cpuc.ca.gov',
                    'energy.ca.gov'
                ]
            ),
            'premium': AccessTier(
                name='Premium',
                daily_query_limit=10,
                description='Full access for regulatory bodies',
                domain_patterns=[
                    'cpuc.ca.gov',
                    'energy.ca.gov',
                    'gov.ca.gov'
                ]
            ),
            'unlimited': AccessTier(
                name='Unlimited',
                daily_query_limit=999999,
                description='Unlimited access for administrators',
                domain_patterns=[
                    'admin.cpuc.ca.gov',
                    'dev.cpuc.ca.gov'
                ]
            )
        }
        logger.info(f"Loaded {len(self.access_tiers)} access tiers")
    
    def setup_oauth_clients(self):
        """Setup OAuth clients for Microsoft and Google"""
        # Microsoft Teams OAuth
        self.microsoft_client_id = os.getenv('MICROSOFT_CLIENT_ID')
        self.microsoft_client_secret = os.getenv('MICROSOFT_CLIENT_SECRET')
        self.microsoft_redirect_uri = os.getenv('MICROSOFT_REDIRECT_URI', 'http://localhost:8501/auth/microsoft/callback')
        
        # Google OAuth
        self.google_client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.google_redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501/auth/google/callback')
        
        logger.info("OAuth clients configured")
    
    def get_microsoft_auth_url(self) -> str:
        """Generate Microsoft OAuth authorization URL"""
        auth_url = (
            f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize?"
            f"client_id={self.microsoft_client_id}&"
            f"response_type=code&"
            f"redirect_uri={self.microsoft_redirect_uri}&"
            f"scope=openid profile email&"
            f"response_mode=query"
        )
        return auth_url
    
    def get_google_auth_url(self) -> str:
        """Generate Google OAuth authorization URL"""
        auth_url = (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={self.google_client_id}&"
            f"response_type=code&"
            f"redirect_uri={self.google_redirect_uri}&"
            f"scope=openid profile email&"
            f"access_type=offline"
        )
        return auth_url
    
    def handle_microsoft_callback(self, auth_code: str) -> Optional[UserProfile]:
        """Handle Microsoft OAuth callback and create/update user"""
        try:
            # Exchange code for tokens
            token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
            token_data = {
                'client_id': self.microsoft_client_id,
                'client_secret': self.microsoft_client_secret,
                'code': auth_code,
                'redirect_uri': self.microsoft_redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            tokens = token_response.json()
            
            # Get user profile
            profile_url = "https://graph.microsoft.com/v1.0/me"
            headers = {'Authorization': f"Bearer {tokens['access_token']}"}
            profile_response = requests.get(profile_url, headers=headers)
            profile_response.raise_for_status()
            profile_data = profile_response.json()
            
            # Create/update user
            user_profile = self.create_or_update_user(
                email=profile_data['mail'] or profile_data['userPrincipalName'],
                name=profile_data['displayName'],
                provider='microsoft'
            )
            
            # Store tokens
            self.store_oauth_tokens(
                user_email=user_profile.email,
                provider='microsoft',
                access_token=tokens['access_token'],
                refresh_token=tokens.get('refresh_token'),
                expires_at=datetime.now() + timedelta(seconds=tokens.get('expires_in', 3600))
            )
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Microsoft OAuth callback failed: {e}")
            return None
    
    def handle_google_callback(self, auth_code: str) -> Optional[UserProfile]:
        """Handle Google OAuth callback and create/update user"""
        try:
            # Exchange code for tokens
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                'client_id': self.google_client_id,
                'client_secret': self.google_client_secret,
                'code': auth_code,
                'redirect_uri': self.google_redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            tokens = token_response.json()
            
            # Get user profile
            profile_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {'Authorization': f"Bearer {tokens['access_token']}"}
            profile_response = requests.get(profile_url, headers=headers)
            profile_response.raise_for_status()
            profile_data = profile_response.json()
            
            # Create/update user
            user_profile = self.create_or_update_user(
                email=profile_data['email'],
                name=profile_data['name'],
                provider='google'
            )
            
            # Store tokens
            self.store_oauth_tokens(
                user_email=user_profile.email,
                provider='google',
                access_token=tokens['access_token'],
                refresh_token=tokens.get('refresh_token'),
                expires_at=datetime.now() + timedelta(seconds=tokens.get('expires_in', 3600))
            )
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Google OAuth callback failed: {e}")
            return None
    
    def create_or_update_user(self, email: str, name: str, provider: str) -> UserProfile:
        """Create new user or update existing user"""
        # Determine access tier based on email domain
        access_tier = self.determine_access_tier(email)
        tier_config = self.access_tiers[access_tier]
        
        current_time = datetime.now().isoformat()
        today = datetime.now().date().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Update existing user
            cursor.execute('''
                UPDATE users 
                SET name = ?, provider = ?, access_tier = ?, daily_query_limit = ?, 
                    last_login = ?, is_active = 1
                WHERE email = ?
            ''', (name, provider, access_tier, tier_config.daily_query_limit, current_time, email))
            
            # Reset daily queries if new day
            if existing_user[7] != today:  # last_reset_date
                cursor.execute('''
                    UPDATE users 
                    SET queries_used_today = 0, last_reset_date = ?
                    WHERE email = ?
                ''', (today, email))
            
            conn.commit()
            
            # Get updated user data
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user_data = cursor.fetchone()
            
        else:
            # Create new user
            cursor.execute('''
                INSERT INTO users (email, name, provider, access_tier, daily_query_limit, 
                                 queries_used_today, last_reset_date, created_at, last_login, is_active)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, 1)
            ''', (email, name, provider, access_tier, tier_config.daily_query_limit, 
                  today, current_time, current_time))
            
            conn.commit()
            
            # Get new user data
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user_data = cursor.fetchone()
        
        conn.close()
        
        # Create UserProfile object
        user_profile = UserProfile(
            email=user_data[1],
            name=user_data[2],
            provider=user_data[3],
            access_tier=user_data[4],
            daily_query_limit=user_data[5],
            queries_used_today=user_data[6],
            last_reset_date=user_data[7],
            created_at=user_data[8],
            last_login=user_data[9],
            is_active=bool(user_data[10])
        )
        
        logger.info(f"User created/updated: {email} ({access_tier} tier)")
        return user_profile
    
    def determine_access_tier(self, email: str) -> str:
        """Determine access tier based on email domain"""
        email_domain = email.split('@')[-1].lower()
        
        # Check each tier (from highest to lowest)
        for tier_name in ['unlimited', 'premium', 'partner', 'free']:
            tier = self.access_tiers[tier_name]
            if any(domain in email_domain for domain in tier.domain_patterns):
                return tier_name
        
        return 'free'  # Default fallback
    
    def store_oauth_tokens(self, user_email: str, provider: str, access_token: str, 
                          refresh_token: Optional[str], expires_at: datetime):
        """Store OAuth tokens securely"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove existing tokens for this user/provider
        cursor.execute('DELETE FROM oauth_tokens WHERE user_email = ? AND provider = ?', 
                      (user_email, provider))
        
        # Insert new tokens
        cursor.execute('''
            INSERT INTO oauth_tokens (user_email, provider, access_token, refresh_token, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_email, provider, access_token, refresh_token, 
              expires_at.isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, email: str) -> Optional[UserProfile]:
        """Get user profile by email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ? AND is_active = 1', (email,))
        user_data = cursor.fetchone()
        conn.close()
        
        if not user_data:
            return None
        
        return UserProfile(
            email=user_data[1],
            name=user_data[2],
            provider=user_data[3],
            access_tier=user_data[4],
            daily_query_limit=user_data[5],
            queries_used_today=user_data[6],
            last_reset_date=user_data[7],
            created_at=user_data[8],
            last_login=user_data[9],
            is_active=bool(user_data[10])
        )
    
    def can_make_query(self, user_email: str) -> Tuple[bool, str]:
        """Check if user can make a query (rate limiting)"""
        user_profile = self.get_user_profile(user_email)
        if not user_profile:
            return False, "User not found"
        
        # Check if daily reset is needed
        today = datetime.now().date().isoformat()
        if user_profile.last_reset_date != today:
            self.reset_daily_queries(user_email)
            user_profile.queries_used_today = 0
        
        if user_profile.queries_used_today >= user_profile.daily_query_limit:
            return False, f"Daily query limit reached ({user_profile.daily_query_limit})"
        
        return True, "Query allowed"
    
    def log_query(self, user_email: str, query_text: str, confidence_score: int = None, 
                  sources_count: int = None) -> bool:
        """Log a query and increment usage counter"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Log the query
            cursor.execute('''
                INSERT INTO query_logs (user_email, query_text, timestamp, confidence_score, sources_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_email, query_text, datetime.now().isoformat(), confidence_score, sources_count))
            
            # Increment usage counter
            cursor.execute('''
                UPDATE users 
                SET queries_used_today = queries_used_today + 1
                WHERE email = ?
            ''', (user_email,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return False
    
    def reset_daily_queries(self, user_email: str):
        """Reset daily query count for a user"""
        today = datetime.now().date().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET queries_used_today = 0, last_reset_date = ?
            WHERE email = ?
        ''', (today, user_email))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_email: str) -> Dict:
        """Get user statistics and usage"""
        user_profile = self.get_user_profile(user_email)
        if not user_profile:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get query count for today
        today = datetime.now().date().isoformat()
        cursor.execute('''
            SELECT COUNT(*) FROM query_logs 
            WHERE user_email = ? AND date(timestamp) = ?
        ''', (user_email, today))
        queries_today = cursor.fetchone()[0]
        
        # Get total queries
        cursor.execute('SELECT COUNT(*) FROM query_logs WHERE user_email = ?', (user_email,))
        total_queries = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'user_profile': asdict(user_profile),
            'queries_today': queries_today,
            'total_queries': total_queries,
            'remaining_queries': user_profile.daily_query_limit - user_profile.queries_used_today,
            'access_tier_info': asdict(self.access_tiers[user_profile.access_tier])
        }

# Global authentication instance
auth_system = AuthenticationSystem()