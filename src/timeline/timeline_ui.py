"""
Timeline UI Components for CPUC RAG System

This module provides Streamlit-based UI components for timeline visualization
and interaction, including date range selection, event filtering, and detail views.

Author: Claude Code
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import pandas as pd
import logging

# Try relative imports first, fall back to absolute
try:
    from .timeline_models import (
        TimelineEvent, TimelineData, EventType, TimelineFilter, ImportanceLevel
    )
    from .timeline_processor import TimelineProcessor
except ImportError:
    from timeline.timeline_models import (
        TimelineEvent, TimelineData, EventType, TimelineFilter, ImportanceLevel
    )
    from timeline.timeline_processor import TimelineProcessor

logger = logging.getLogger(__name__)


class TimelineUI:
    """Main UI component for timeline visualization"""
    
    def __init__(self, timeline_processor: TimelineProcessor):
        """Initialize timeline UI with processor"""
        self.processor = timeline_processor
        self.colors = self._get_event_colors()
    
    def _get_event_colors(self) -> Dict[EventType, str]:
        """Define color scheme for different event types"""
        return {
            EventType.DECISION: "#FF6B6B",           # Red - Critical
            EventType.RULING: "#4ECDC4",             # Teal - Important
            EventType.PROPOSED_DECISION: "#45B7D1",  # Blue - Important
            EventType.MOTION: "#96CEB4",             # Green - Medium
            EventType.BRIEF: "#FFEAA7",              # Yellow - Medium
            EventType.TESTIMONY: "#DDA0DD",          # Plum - Medium
            EventType.COMMENT: "#A8E6CF",            # Light Green - Low
            EventType.REPLY: "#FFD93D",              # Light Yellow - Low
            EventType.NOTICE: "#C7CEEA",             # Light Blue - Info
            EventType.APPLICATION: "#FFB6C1",        # Light Pink - Medium
            EventType.AMENDMENT: "#F4A460",          # Sandy Brown - Medium
            EventType.EXPARTE: "#DEB887",            # Burlywood - Medium
            EventType.COMPLIANCE: "#D3D3D3",         # Light Gray - Info
            EventType.RESPONSE: "#98FB98",           # Pale Green - Low
            EventType.STATEMENT: "#E6E6FA",          # Lavender - Info
            EventType.REQUEST: "#F0E68C",            # Khaki - Low
            EventType.OTHER: "#BEBEBE"               # Gray - Other
        }
    
    def render_timeline_page(self, proceeding_number: str = None):
        """Render the main timeline page"""
        st.title("📅 CPUC Proceeding Timeline")
        
        # Show loading while extracting timeline data
        with st.spinner("Loading timeline data..."):
            timeline_data = self.processor.extract_timeline_events(proceeding_number)
        
        if not timeline_data.events:
            st.warning("No timeline events found. Please check if the vector store has been built.")
            return
        
        # Show timeline summary
        self._render_timeline_summary(timeline_data)
        
        # Render controls
        filter_criteria = self._render_timeline_controls(timeline_data)
        
        # Filter events based on controls
        filtered_timeline = self.processor.filter_timeline_events(timeline_data, filter_criteria)
        
        if not filtered_timeline.events:
            st.warning("No events match the selected filters.")
            return
        
        # Render main timeline visualization
        self._render_timeline_chart(filtered_timeline)
        
        # Render event details
        self._render_event_details(filtered_timeline)
    
    def _render_timeline_summary(self, timeline_data: TimelineData):
        """Render timeline summary statistics"""
        st.subheader("📊 Timeline Overview")
        
        summary = self.processor.get_timeline_summary(timeline_data)
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", summary["total_events"])
        
        with col2:
            if summary["date_range"] and summary["date_range"][0]:
                start_date = summary["date_range"][0]
                end_date = summary["date_range"][1]
                duration = (end_date - start_date).days
                st.metric("Timeline Span", f"{duration} days")
            else:
                st.metric("Timeline Span", "N/A")
        
        with col3:
            major_events = len([e for e in timeline_data.events if e.importance_score >= 7])
            st.metric("Major Events", major_events)
        
        with col4:
            st.metric("Participants", len(summary["participants"]))
        
        # Show event type distribution
        if summary["event_types"]:
            st.subheader("📋 Event Type Distribution")
            
            # Create pie chart for event types
            event_types_df = pd.DataFrame(
                list(summary["event_types"].items()),
                columns=["Event Type", "Count"]
            )
            
            fig = px.pie(
                event_types_df, 
                values="Count", 
                names="Event Type",
                title="Events by Type"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_timeline_controls(self, timeline_data: TimelineData) -> TimelineFilter:
        """Render timeline control widgets and return filter criteria"""
        st.subheader("🎛️ Timeline Filters")
        
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selector
            if timeline_data.date_range[0]:
                min_date = timeline_data.date_range[0].date()
                max_date = timeline_data.date_range[1].date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select date range to filter events"
                )
                
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date = datetime.combine(date_range[0], datetime.min.time())
                    end_date = datetime.combine(date_range[1], datetime.max.time())
                else:
                    start_date = None
                    end_date = None
            else:
                start_date = None
                end_date = None
                st.info("No date range available")
            
            # Event type selector
            available_types = list(timeline_data.event_types)
            selected_types = st.multiselect(
                "Event Types",
                options=available_types,
                default=available_types,
                format_func=lambda x: x.value,
                help="Select event types to display"
            )
        
        with col2:
            # Importance range selector
            importance_range = st.slider(
                "Importance Range",
                min_value=1,
                max_value=10,
                value=(1, 10),
                help="Filter events by importance score"
            )
            
            # Participant selector
            available_participants = list(timeline_data.participants)
            if available_participants:
                selected_participants = st.multiselect(
                    "Participants",
                    options=available_participants,
                    default=[],
                    help="Filter by participating organizations"
                )
            else:
                selected_participants = []
        
        # Create filter criteria
        filter_criteria = TimelineFilter(
            start_date=start_date,
            end_date=end_date,
            event_types=set(selected_types),
            min_importance=importance_range[0],
            max_importance=importance_range[1],
            participants=set(selected_participants)
        )
        
        return filter_criteria
    
    def _render_timeline_chart(self, timeline_data: TimelineData):
        """Render interactive timeline chart"""
        st.subheader("📈 Timeline Visualization")
        
        if not timeline_data.events:
            st.warning("No events to display")
            return
        
        # Sort events by date
        events = timeline_data.sort_events(by_date=True)
        
        # Create data for plotly
        dates = [event.event_date for event in events]
        titles = [event.title for event in events]
        types = [event.event_type.value for event in events]
        colors = [self.colors[event.event_type] for event in events]
        importance = [event.importance_score for event in events]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points for each event type
        for event_type in set(types):
            event_indices = [i for i, t in enumerate(types) if t == event_type]
            fig.add_trace(go.Scatter(
                x=[dates[i] for i in event_indices],
                y=[importance[i] for i in event_indices],
                mode='markers',
                name=event_type,
                marker=dict(
                    size=10,
                    color=self.colors[EventType(event_type)],
                    symbol='circle'
                ),
                text=[titles[i] for i in event_indices],
                hovertemplate='<b>%{text}</b><br>' +
                             'Date: %{x}<br>' +
                             'Importance: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="Timeline of CPUC Proceeding Events",
            xaxis_title="Date",
            yaxis_title="Importance Score",
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        # Add importance level annotations
        fig.add_hline(y=7, line_dash="dash", line_color="red", 
                     annotation_text="Major Events", annotation_position="bottom right")
        fig.add_hline(y=4, line_dash="dash", line_color="orange", 
                     annotation_text="Standard Events", annotation_position="bottom right")
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_event_details(self, timeline_data: TimelineData):
        """Render detailed event list with click-to-expand functionality"""
        st.subheader("📋 Event Details")
        
        if not timeline_data.events:
            return
        
        # Sort events by date (most recent first)
        events = timeline_data.sort_events(by_date=True, ascending=False)
        
        # Group events by month for better organization
        events_by_month = {}
        for event in events:
            month_key = event.event_date.strftime("%Y-%m")
            if month_key not in events_by_month:
                events_by_month[month_key] = []
            events_by_month[month_key].append(event)
        
        # Display events by month
        for month_key in sorted(events_by_month.keys(), reverse=True):
            month_events = events_by_month[month_key]
            month_name = datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
            
            with st.expander(f"📅 {month_name} ({len(month_events)} events)", expanded=False):
                for event in month_events:
                    self._render_event_card(event)
    
    def _render_event_card(self, event: TimelineEvent):
        """Render individual event card"""
        # Create importance indicator
        importance_color = self._get_importance_color(event.importance_score)
        
        # Create event card
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            
            with col1:
                st.markdown(f"<div style='background-color: {importance_color}; "
                           f"width: 20px; height: 20px; border-radius: 50%; "
                           f"display: inline-block; margin-right: 10px;'></div>",
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{event.title}**")
                st.markdown(f"*{event.event_type.value}* • {event.event_date.strftime('%B %d, %Y')}")
                
                if event.description:
                    st.markdown(f"<small>{event.description}</small>", unsafe_allow_html=True)
                
                # Show participants
                if event.participants:
                    participants_str = ", ".join(event.participants)
                    st.markdown(f"<small>**Participants:** {participants_str}</small>", 
                               unsafe_allow_html=True)
            
            with col3:
                # Add link to source document
                if event.source_url:
                    st.markdown(f"[📄 View Document]({event.source_url})", 
                               unsafe_allow_html=True)
                
                # Show importance score
                st.markdown(f"<small>Score: {event.importance_score}/10</small>", 
                           unsafe_allow_html=True)
            
            st.markdown("---")
    
    def _get_importance_color(self, score: int) -> str:
        """Get color based on importance score"""
        if score >= 9:
            return "#FF0000"  # Red - Critical
        elif score >= 7:
            return "#FFA500"  # Orange - High
        elif score >= 5:
            return "#FFD700"  # Gold - Medium
        elif score >= 3:
            return "#90EE90"  # Light Green - Low
        else:
            return "#D3D3D3"  # Light Gray - Informational
    
    def render_timeline_export(self, timeline_data: TimelineData):
        """Render timeline export functionality"""
        st.subheader("💾 Export Timeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                json_data = self.processor.export_timeline_data(timeline_data, "json")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as CSV"):
                csv_data = self.processor.export_timeline_data(timeline_data, "csv")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def render_timeline_search(self, timeline_data: TimelineData) -> List[TimelineEvent]:
        """Render timeline search functionality"""
        st.subheader("🔍 Search Timeline")
        
        search_query = st.text_input(
            "Search events",
            placeholder="Enter keywords to search in titles and descriptions",
            help="Search within event titles and descriptions"
        )
        
        if search_query:
            # Filter events by search query
            filtered_events = []
            query_lower = search_query.lower()
            
            for event in timeline_data.events:
                if (query_lower in event.title.lower() or 
                    query_lower in event.description.lower() or
                    any(query_lower in keyword.lower() for keyword in event.keywords)):
                    filtered_events.append(event)
            
            st.info(f"Found {len(filtered_events)} events matching '{search_query}'")
            return filtered_events
        
        return timeline_data.events


def create_timeline_ui(timeline_processor: TimelineProcessor) -> TimelineUI:
    """Factory function to create timeline UI"""
    return TimelineUI(timeline_processor)