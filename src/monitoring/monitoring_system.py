#!/usr/bin/env python3
"""
Comprehensive Monitoring and Logging System

Provides production-grade monitoring, performance tracking, and logging
capabilities for the CPUC RAG system with real-time metrics and alerts.

Author: Claude Code
"""

import gc
import json
import logging
import os
import psutil
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

import config

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.PROJECT_ROOT / 'cpuc_rag.log')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: str
    operation: str
    duration: float
    success: bool
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class ResourceSnapshot:
    """System resource snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    gpu_percent: float = 0.0
    gpu_memory_gb: float = 0.0


@dataclass
class ErrorEvent:
    """Error event data structure."""
    timestamp: str
    level: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]


class PerformanceMonitor:
    """Real-time performance monitoring with metrics collection."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.operation_stats = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()
        
    def start_operation(self, operation_id: str, operation_name: str):
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation_id] = {
                'name': operation_name,
                'start_time': time.time(),
                'start_resources': self._get_resource_snapshot()
            }
    
    def end_operation(self, operation_id: str, success: bool = True, metadata: Dict[str, Any] = None):
        """End timing an operation and record metrics."""
        with self.lock:
            if operation_id not in self.start_times:
                logger.warning(f"Operation {operation_id} not found in start times")
                return
            
            start_info = self.start_times.pop(operation_id)
            end_time = time.time()
            duration = end_time - start_info['start_time']
            end_resources = self._get_resource_snapshot()
            
            # Calculate resource usage delta
            resource_delta = {
                'cpu_change': end_resources.cpu_percent - start_info['start_resources'].cpu_percent,
                'memory_change': end_resources.memory_percent - start_info['start_resources'].memory_percent,
                'gpu_change': end_resources.gpu_percent - start_info['start_resources'].gpu_percent
            }
            
            # Create performance metric
            metric = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                operation=start_info['name'],
                duration=duration,
                success=success,
                resource_usage=resource_delta,
                metadata=metadata or {}
            )
            
            # Store metrics
            self.metrics_history.append(metric)
            self.operation_stats[start_info['name']].append(metric)
            
            # Log performance if enabled
            if config.PERFORMANCE_LOGGING_ENABLED:
                self._log_performance_metric(metric)
    
    def _get_resource_snapshot(self) -> ResourceSnapshot:
        """Get current system resource snapshot."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        gpu_percent = 0.0
        gpu_memory_gb = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = (gpu_memory / gpu_total) * 100
                gpu_memory_gb = gpu_memory / (1024**3)
        except ImportError:
            pass
        
        return ResourceSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            gpu_percent=gpu_percent,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def _log_performance_metric(self, metric: PerformanceMetrics):
        """Log performance metric."""
        logger.info(
            f"PERF: {metric.operation} completed in {metric.duration:.2f}s "
            f"(success: {metric.success})"
        )
        
        if metric.duration > 60:  # Log slow operations
            logger.warning(
                f"SLOW: {metric.operation} took {metric.duration:.2f}s "
                f"(threshold: 60s)"
            )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.operation_stats:
            return {'error': 'Operation not found'}
        
        metrics = self.operation_stats[operation_name]
        durations = [m.duration for m in metrics]
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        return {
            'operation': operation_name,
            'total_runs': len(metrics),
            'success_rate': success_rate,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'recent_runs': len([m for m in metrics if 
                              datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(hours=1)])
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        current_resources = self._get_resource_snapshot()
        
        # Health scoring
        health_score = 100
        alerts = []
        
        if current_resources.cpu_percent > 90:
            health_score -= 20
            alerts.append("HIGH_CPU")
        elif current_resources.cpu_percent > 75:
            health_score -= 10
            alerts.append("ELEVATED_CPU")
        
        if current_resources.memory_percent > 90:
            health_score -= 25
            alerts.append("HIGH_MEMORY")
        elif current_resources.memory_percent > 80:
            health_score -= 15
            alerts.append("ELEVATED_MEMORY")
        
        if current_resources.disk_percent > 95:
            health_score -= 30
            alerts.append("CRITICAL_DISK")
        elif current_resources.disk_percent > 85:
            health_score -= 20
            alerts.append("HIGH_DISK")
        
        if current_resources.gpu_percent > 95:
            health_score -= 20
            alerts.append("HIGH_GPU")
        elif current_resources.gpu_percent > 85:
            health_score -= 10
            alerts.append("ELEVATED_GPU")
        
        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'alerts': alerts,
            'resources': asdict(current_resources),
            'timestamp': current_resources.timestamp
        }


class ErrorTracker:
    """Error tracking and analytics system."""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.error_history = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.lock = threading.Lock()
        
    def log_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Log an error event."""
        import traceback
        
        with self.lock:
            error_event = ErrorEvent(
                timestamp=datetime.now().isoformat(),
                level='ERROR',
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                context=context or {}
            )
            
            self.error_history.append(error_event)
            self.error_counts[error_event.error_type] += 1
            
            # Log to standard logger
            logger.error(
                f"ERROR in {operation}: {error_event.error_type} - {error_event.error_message}",
                extra={'context': context}
            )
    
    def log_warning(self, operation: str, message: str, context: Dict[str, Any] = None):
        """Log a warning event."""
        with self.lock:
            warning_event = ErrorEvent(
                timestamp=datetime.now().isoformat(),
                level='WARNING',
                operation=operation,
                error_type='Warning',
                error_message=message,
                stack_trace=None,
                context=context or {}
            )
            
            self.error_history.append(warning_event)
            logger.warning(f"WARNING in {operation}: {message}", extra={'context': context})
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error.timestamp) > cutoff_time
        ]
        
        error_types = defaultdict(int)
        operations = defaultdict(int)
        
        for error in recent_errors:
            error_types[error.error_type] += 1
            operations[error.operation] += 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': dict(error_types),
            'affected_operations': dict(operations),
            'error_rate': len(recent_errors) / max(hours, 1),  # Errors per hour
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }


class MonitoringSystem:
    """Comprehensive monitoring system coordinator."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.error_tracker = ErrorTracker()
        self.resource_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
    def start_monitoring(self, interval: int = 60):
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect resource snapshot
                snapshot = self.performance_monitor._get_resource_snapshot()
                self.resource_history.append(snapshot)
                
                # Check for alerts
                health = self.performance_monitor.get_system_health()
                if health['alerts']:
                    self._trigger_alerts(health)
                
                # Periodic cleanup
                if len(self.resource_history) % 60 == 0:  # Every hour
                    self._periodic_cleanup()
                
                time.sleep(interval)
                
            except Exception as e:
                self.error_tracker.log_error("monitoring_loop", e)
                time.sleep(interval)
    
    def _trigger_alerts(self, health: Dict[str, Any]):
        """Trigger alerts for system health issues."""
        for alert_type in health['alerts']:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, health)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _periodic_cleanup(self):
        """Perform periodic cleanup tasks."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("Periodic cleanup completed")
            
        except Exception as e:
            self.error_tracker.log_error("periodic_cleanup", e)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health = self.performance_monitor.get_system_health()
        error_summary = self.error_tracker.get_error_summary()
        
        # Resource trends
        if len(self.resource_history) > 1:
            recent_resources = list(self.resource_history)[-60:]  # Last hour
            cpu_trend = sum(r.cpu_percent for r in recent_resources) / len(recent_resources)
            memory_trend = sum(r.memory_percent for r in recent_resources) / len(recent_resources)
            gpu_trend = sum(r.gpu_percent for r in recent_resources) / len(recent_resources)
        else:
            cpu_trend = memory_trend = gpu_trend = 0
        
        # Operation performance
        common_operations = ['document_processing', 'embedding_generation', 'vector_search']
        operation_performance = {}
        
        for op in common_operations:
            stats = self.performance_monitor.get_operation_stats(op)
            if 'error' not in stats:
                operation_performance[op] = stats
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': health,
            'error_summary': error_summary,
            'resource_trends': {
                'cpu_avg_1h': cpu_trend,
                'memory_avg_1h': memory_trend,
                'gpu_avg_1h': gpu_trend
            },
            'operation_performance': operation_performance,
            'monitoring_status': {
                'active': self.monitoring_active,
                'resource_history_size': len(self.resource_history),
                'performance_metrics_size': len(self.performance_monitor.metrics_history),
                'error_history_size': len(self.error_tracker.error_history)
            }
        }
    
    def export_monitoring_data(self, output_dir: Path = None) -> Path:
        """Export monitoring data to files."""
        if output_dir is None:
            output_dir = config.PROJECT_ROOT / 'monitoring_exports'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export report
        report = self.generate_monitoring_report()
        report_file = output_dir / f'monitoring_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Export raw metrics
        metrics_data = [asdict(m) for m in self.performance_monitor.metrics_history]
        metrics_file = output_dir / f'performance_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Export error data
        error_data = [asdict(e) for e in self.error_tracker.error_history]
        error_file = output_dir / f'error_history_{timestamp}.json'
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        # Export resource history
        resource_data = [asdict(r) for r in self.resource_history]
        resource_file = output_dir / f'resource_history_{timestamp}.json'
        with open(resource_file, 'w') as f:
            json.dump(resource_data, f, indent=2)
        
        logger.info(f"Monitoring data exported to {output_dir}")
        return output_dir


# Global monitoring instance
global_monitor = MonitoringSystem()


# Convenience functions and decorators

def monitor_operation(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            operation_id = f"{operation_name}_{id(func)}_{time.time()}"
            global_monitor.performance_monitor.start_operation(operation_id, operation_name)
            
            try:
                result = func(*args, **kwargs)
                global_monitor.performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                global_monitor.performance_monitor.end_operation(operation_id, success=False)
                global_monitor.error_tracker.log_error(operation_name, e)
                raise
        
        return wrapper
    return decorator


def log_performance(operation: str, duration: float, success: bool = True, metadata: Dict[str, Any] = None):
    """Log a performance metric directly."""
    # Create a temporary operation to log
    operation_id = f"manual_{operation}_{time.time()}"
    
    # Simulate start time
    global_monitor.performance_monitor.start_times[operation_id] = {
        'name': operation,
        'start_time': time.time() - duration,
        'start_resources': global_monitor.performance_monitor._get_resource_snapshot()
    }
    
    global_monitor.performance_monitor.end_operation(operation_id, success, metadata)


def log_error(operation: str, error: Exception, context: Dict[str, Any] = None):
    """Log an error event."""
    global_monitor.error_tracker.log_error(operation, error, context)


def log_warning(operation: str, message: str, context: Dict[str, Any] = None):
    """Log a warning event."""
    global_monitor.error_tracker.log_warning(operation, message, context)


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return global_monitor.performance_monitor.get_system_health()


def print_monitoring_summary():
    """Print a formatted monitoring summary."""
    report = global_monitor.generate_monitoring_report()
    
    print("\n📊 SYSTEM MONITORING SUMMARY")
    print("=" * 50)
    
    # System Health
    health = report['system_health']
    health_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'warning' else "🚨"
    print(f"{health_icon} System Health: {health['status'].upper()} (Score: {health['health_score']}/100)")
    
    if health['alerts']:
        print(f"🚨 Active Alerts: {', '.join(health['alerts'])}")
    
    # Resource Status
    resources = health['resources']
    print(f"\n💾 Resources:")
    print(f"   CPU: {resources['cpu_percent']:.1f}%")
    print(f"   Memory: {resources['memory_percent']:.1f}% ({resources['memory_available_gb']:.1f}GB available)")
    print(f"   Disk: {resources['disk_percent']:.1f}% ({resources['disk_free_gb']:.1f}GB free)")
    if resources['gpu_percent'] > 0:
        print(f"   GPU: {resources['gpu_percent']:.1f}% ({resources['gpu_memory_gb']:.1f}GB used)")
    
    # Error Summary
    errors = report['error_summary']
    if errors['total_errors'] > 0:
        print(f"\n❌ Errors (24h): {errors['total_errors']} ({errors['error_rate']:.1f}/hour)")
        if errors['most_common_error']:
            print(f"   Most common: {errors['most_common_error'][0]} ({errors['most_common_error'][1]} occurrences)")
    else:
        print(f"\n✅ No errors in the last 24 hours")
    
    # Performance Summary
    performance = report['operation_performance']
    if performance:
        print(f"\n⚡ Performance:")
        for op_name, stats in performance.items():
            print(f"   {op_name}: {stats['avg_duration']:.2f}s avg, {stats['success_rate']:.1%} success")


# Initialize monitoring if enabled
if config.RESOURCE_MONITORING_ENABLED:
    global_monitor.start_monitoring(config.RESOURCE_LOG_INTERVAL)
    logger.info("Monitoring system initialized and started")


if __name__ == "__main__":
    print("🔍 CPUC RAG MONITORING SYSTEM")
    print("=" * 40)
    
    # Start monitoring
    global_monitor.start_monitoring(interval=5)
    
    # Test operations
    @monitor_operation("test_operation")
    def test_function():
        time.sleep(1)
        return "success"
    
    # Run test
    print("Running test operation...")
    result = test_function()
    
    # Wait for data collection
    time.sleep(6)
    
    # Show summary
    print_monitoring_summary()
    
    # Export data
    export_path = global_monitor.export_monitoring_data()
    print(f"\n📁 Data exported to: {export_path}")
    
    # Stop monitoring
    global_monitor.stop_monitoring()
    print("\n✅ Monitoring test completed")