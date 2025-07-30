"""
Timeline Integration for CPUC RAG System

This module integrates timeline functionality with the existing RAG system,
providing seamless access to timeline data and ensuring synchronization
with vector store updates.

Author: Claude Code
"""

import logging
from typing import Optional, Dict, List
import streamlit as st
from datetime import datetime

# Try relative imports first, fall back to absolute
try:
    from .timeline_processor import TimelineProcessor, create_timeline_processor
    from .timeline_ui import TimelineUI, create_timeline_ui
    from .timeline_models import TimelineData, TimelineEvent, EventType, TimelineFilter
    from ..core.rag_core import CPUCRAGSystem
except ImportError:
    # Fallback to absolute imports from src/
    from timeline.timeline_processor import TimelineProcessor, create_timeline_processor
    from timeline.timeline_ui import TimelineUI, create_timeline_ui
    from timeline.timeline_models import TimelineData, TimelineEvent, EventType, TimelineFilter
    from core.rag_core import CPUCRAGSystem

logger = logging.getLogger(__name__)


class TimelineIntegration:
    """Main integration class for timeline functionality"""
    
    def __init__(self, rag_system: CPUCRAGSystem):
        """Initialize timeline integration with RAG system"""
        self.rag_system = rag_system
        self.processor = create_timeline_processor(rag_system)
        self.ui = create_timeline_ui(self.processor)
        self._timeline_cache = {}
        self._last_refresh = None
    
    def initialize_timeline_system(self) -> bool:
        """Initialize timeline system and verify functionality"""
        try:
            logger.info("Initializing timeline system...")
            
            # Verify RAG system is available
            if not self.rag_system or not hasattr(self.rag_system, 'vectordb'):
                logger.error("RAG system not available or missing vector database")
                return False
            
            # Check if vector store is loaded
            if self.rag_system.vectordb is None:
                logger.warning("Vector store not loaded - timeline will be empty until vector store is built")
                return True  # Return True but with empty timeline
            
            # Test timeline extraction
            test_timeline = self.processor.extract_timeline_events()
            if not test_timeline.events:
                logger.warning("No timeline events found - vector store may be empty")
                return True  # Return True but with empty timeline
            
            logger.info(f"Timeline system initialized with {len(test_timeline.events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize timeline system: {e}")
            return False
    
    def get_timeline_data(self, proceeding_number: str = None, 
                         refresh: bool = False) -> Optional[TimelineData]:
        """Get timeline data for specified proceeding"""
        try:
            return self.processor.extract_timeline_events(proceeding_number, refresh)
        except Exception as e:
            logger.error(f"Failed to get timeline data: {e}")
            return None
    
    def render_timeline_interface(self, proceeding_number: str = None):
        """Render complete timeline interface"""
        try:
            # Check if vector store is available
            if self.rag_system.vectordb is None:
                st.info("📊 Vector store is currently being built. Timeline will be available once processing is complete.")
                st.info("🔄 This may take a few minutes. Please refresh the page or wait for the system to complete initialization.")
                return
            
            # Check if timeline system is initialized
            if not self.initialize_timeline_system():
                st.error("Timeline system could not be initialized. Please check if the vector store is built.")
                return
            
            # Render timeline page
            self.ui.render_timeline_page(proceeding_number)
            
        except Exception as e:
            logger.error(f"Error rendering timeline interface: {e}")
            st.error(f"Error displaying timeline: {e}")
    
    def get_events_for_query_context(self, query: str, max_events: int = 5) -> List[TimelineEvent]:
        """Get relevant timeline events for query context"""
        try:
            timeline_data = self.get_timeline_data()
            if not timeline_data:
                return []
            
            # Find events relevant to query
            relevant_events = []
            query_lower = query.lower()
            
            for event in timeline_data.events:
                # Check if event is relevant to query
                if (query_lower in event.title.lower() or 
                    query_lower in event.description.lower() or
                    any(query_lower in keyword.lower() for keyword in event.keywords)):
                    relevant_events.append(event)
            
            # Sort by importance and date
            relevant_events.sort(key=lambda e: (e.importance_score, e.event_date), reverse=True)
            
            return relevant_events[:max_events]
            
        except Exception as e:
            logger.error(f"Error getting events for query context: {e}")
            return []
    
    def get_recent_major_events(self, days_back: int = 30, min_importance: int = 7) -> List[TimelineEvent]:
        """Get recent major events"""
        try:
            timeline_data = self.get_timeline_data()
            if not timeline_data:
                return []
            
            # Calculate date threshold
            from datetime import timedelta
            threshold_date = datetime.now() - timedelta(days=days_back)
            
            # Filter recent major events
            recent_events = []
            for event in timeline_data.events:
                if (event.event_date >= threshold_date and 
                    event.importance_score >= min_importance):
                    recent_events.append(event)
            
            # Sort by date (most recent first)
            recent_events.sort(key=lambda e: e.event_date, reverse=True)
            
            return recent_events
            
        except Exception as e:
            logger.error(f"Error getting recent major events: {e}")
            return []
    
    def create_timeline_summary_for_rag(self, proceeding_number: str = None) -> str:
        """Create timeline summary for RAG system context"""
        try:
            timeline_data = self.get_timeline_data(proceeding_number)
            if not timeline_data or not timeline_data.events:
                return "No timeline data available."
            
            # Get major events
            major_events = [e for e in timeline_data.events if e.importance_score >= 7]
            major_events.sort(key=lambda e: e.event_date)
            
            # Create summary
            summary_parts = []
            summary_parts.append(f"Timeline Summary for {proceeding_number or 'All Proceedings'}:")
            summary_parts.append(f"- Total Events: {len(timeline_data.events)}")
            summary_parts.append(f"- Major Events: {len(major_events)}")
            
            if timeline_data.date_range[0]:
                summary_parts.append(f"- Date Range: {timeline_data.date_range[0].strftime('%Y-%m-%d')} to {timeline_data.date_range[1].strftime('%Y-%m-%d')}")
            
            if major_events:
                summary_parts.append("\nMajor Events:")
                for event in major_events[-5:]:  # Last 5 major events
                    summary_parts.append(f"- {event.event_date.strftime('%Y-%m-%d')}: {event.title}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error creating timeline summary: {e}")
            return "Error generating timeline summary."
    
    def refresh_timeline_cache(self):
        """Refresh timeline cache"""
        try:
            logger.info("Refreshing timeline cache...")
            self.processor.clear_cache()
            self._timeline_cache.clear()
            self._last_refresh = datetime.now()
            logger.info("Timeline cache refreshed")
        except Exception as e:
            logger.error(f"Error refreshing timeline cache: {e}")
    
    def get_timeline_stats(self) -> Dict:
        """Get timeline statistics"""
        try:
            timeline_data = self.get_timeline_data()
            if not timeline_data:
                return {"error": "No timeline data available"}
            
            return self.processor.get_timeline_summary(timeline_data)
            
        except Exception as e:
            logger.error(f"Error getting timeline stats: {e}")
            return {"error": f"Error getting timeline stats: {e}"}
    
    def search_timeline_events(self, query: str, proceeding_number: str = None) -> List[TimelineEvent]:
        """Search timeline events"""
        try:
            timeline_data = self.get_timeline_data(proceeding_number)
            if not timeline_data:
                return []
            
            # Search events
            results = []
            query_lower = query.lower()
            
            for event in timeline_data.events:
                score = 0
                
                # Check title
                if query_lower in event.title.lower():
                    score += 3
                
                # Check description
                if query_lower in event.description.lower():
                    score += 2
                
                # Check keywords
                for keyword in event.keywords:
                    if query_lower in keyword.lower():
                        score += 1
                
                # Check participants
                for participant in event.participants:
                    if query_lower in participant.lower():
                        score += 1
                
                if score > 0:
                    results.append((event, score))
            
            # Sort by score and importance
            results.sort(key=lambda x: (x[1], x[0].importance_score), reverse=True)
            
            return [event for event, score in results]
            
        except Exception as e:
            logger.error(f"Error searching timeline events: {e}")
            return []
    
    def get_event_neighbors(self, event: TimelineEvent, days_window: int = 30) -> List[TimelineEvent]:
        """Get events that occurred near a specific event"""
        try:
            timeline_data = self.get_timeline_data(event.proceeding_number)
            if not timeline_data:
                return []
            
            from datetime import timedelta
            
            # Calculate date window
            start_date = event.event_date - timedelta(days=days_window)
            end_date = event.event_date + timedelta(days=days_window)
            
            # Find neighboring events
            neighbors = []
            for other_event in timeline_data.events:
                if (other_event != event and 
                    start_date <= other_event.event_date <= end_date):
                    neighbors.append(other_event)
            
            # Sort by date proximity
            neighbors.sort(key=lambda e: abs((e.event_date - event.event_date).days))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting event neighbors: {e}")
            return []
    
    def export_timeline_for_proceeding(self, proceeding_number: str, format: str = "json") -> str:
        """Export timeline data for specific proceeding"""
        try:
            timeline_data = self.get_timeline_data(proceeding_number)
            if not timeline_data:
                return ""
            
            return self.processor.export_timeline_data(timeline_data, format)
            
        except Exception as e:
            logger.error(f"Error exporting timeline: {e}")
            return ""


def create_timeline_integration(rag_system: CPUCRAGSystem) -> TimelineIntegration:
    """Factory function to create timeline integration"""
    return TimelineIntegration(rag_system)