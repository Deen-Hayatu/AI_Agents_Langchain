"""
Module 7: Best Practices & Troubleshooting
==========================================

This module covers production-ready best practices, debugging techniques,
performance optimization, and troubleshooting common issues in AI agent development.

Learning Objectives:
- Implement robust error handling and validation
- Master debugging and monitoring techniques  
- Apply performance optimization strategies
- Handle production deployment challenges
- Troubleshoot common agent issues
"""

import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import json


# =============================================================================
# 1. Error Handling and Validation Framework
# =============================================================================

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentError:
    """Represents an error that occurred during agent execution."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    error_type: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolution_suggestions: List[str] = field(default_factory=list)


class AgentErrorHandler:
    """
    Comprehensive error handling system for AI agents.
    Provides structured error logging, recovery mechanisms, and monitoring.
    """
    
    def __init__(self):
        self.error_log: List[AgentError] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for agent operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent_operations.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AgentSystem')
    
    def handle_error(self, error: Exception, component: str, context: Dict[str, Any] = None) -> AgentError:
        """
        Handle and log errors with structured information.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            context: Additional context information
            
        Returns:
            Structured AgentError object
        """
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_log)}"
        
        # Determine severity
        severity = self._determine_severity(error, component)
        
        # Create structured error
        agent_error = AgentError(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            context=context or {},
            stack_trace=traceback.format_exc(),
            resolution_suggestions=self._get_resolution_suggestions(error, component)
        )
        
        # Log error
        self.error_log.append(agent_error)
        self._log_error(agent_error)
        
        # Attempt recovery if possible
        if severity != ErrorSeverity.CRITICAL:
            self._attempt_recovery(agent_error)
        
        return agent_error
    
    def _determine_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on type and component."""
        critical_components = ["llm_interface", "core_agent", "safety_filter"]
        critical_errors = [ConnectionError, TimeoutError, MemoryError]
        
        if component in critical_components or type(error) in critical_errors:
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_resolution_suggestions(self, error: Exception, component: str) -> List[str]:
        """Generate resolution suggestions based on error type."""
        suggestions = []
        
        if isinstance(error, ConnectionError):
            suggestions.extend([
                "Check network connectivity",
                "Verify API endpoints are accessible",
                "Implement retry mechanism with exponential backoff"
            ])
        elif isinstance(error, TimeoutError):
            suggestions.extend([
                "Increase timeout values",
                "Optimize query complexity",
                "Implement request queuing"
            ])
        elif isinstance(error, KeyError):
            suggestions.extend([
                "Validate input data structure",
                "Add default value handling",
                "Implement schema validation"
            ])
        elif isinstance(error, ValueError):
            suggestions.extend([
                "Validate input parameters",
                "Add type checking",
                "Implement input sanitization"
            ])
        else:
            suggestions.append("Review error context and stack trace for specific guidance")
        
        return suggestions
    
    def _log_error(self, error: AgentError):
        """Log error with appropriate level."""
        log_message = f"[{error.error_id}] {error.component}: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_recovery(self, error: AgentError):
        """Attempt automatic recovery based on error type."""
        recovery_key = f"{error.component}_{error.error_type}"
        
        if recovery_key in self.recovery_strategies:
            try:
                self.recovery_strategies[recovery_key](error)
                self.logger.info(f"Recovery attempted for {error.error_id}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error.error_id}: {recovery_error}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_log:
            return {"message": "No errors recorded"}
        
        total_errors = len(self.error_log)
        severity_counts = {}
        component_counts = {}
        error_type_counts = {}
        
        for error in self.error_log:
            # Count by severity
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            # Count by component
            component_counts[error.component] = component_counts.get(error.component, 0) + 1
            
            # Count by error type
            error_type_counts[error.error_type] = error_type_counts.get(error.error_type, 0) + 1
        
        # Calculate error rate over time
        recent_errors = [e for e in self.error_log if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            "total_errors": total_errors,
            "recent_errors_24h": len(recent_errors),
            "error_rate_24h": len(recent_errors) / 24,  # errors per hour
            "severity_distribution": severity_counts,
            "component_distribution": component_counts,
            "error_type_distribution": error_type_counts,
            "most_problematic_component": max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None
        }


# =============================================================================
# 2. Performance Monitoring and Optimization
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Represents performance metrics for agent operations."""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_usage_mb: float
    tokens_used: int
    api_calls_made: int
    cache_hits: int
    cache_misses: int
    success: bool


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for AI agents.
    Tracks execution times, resource usage, and optimization opportunities.
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.performance_thresholds = {
            "max_duration_seconds": 30.0,
            "max_memory_mb": 500.0,
            "max_tokens_per_request": 4000,
            "min_cache_hit_rate": 0.6
        }
        self.optimization_cache: Dict[str, Any] = {}
    
    def monitor_operation(self, operation_name: str):
        """
        Decorator to monitor agent operation performance.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start monitoring
                start_time = datetime.now()
                tokens_before = self._get_token_count()
                
                try:
                    # Execute operation
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    raise e
                finally:
                    # Record metrics
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        memory_usage_mb=self._get_memory_usage(),
                        tokens_used=self._get_token_count() - tokens_before,
                        api_calls_made=1,  # Simplified
                        cache_hits=self._get_cache_stats()["hits"],
                        cache_misses=self._get_cache_stats()["misses"],
                        success=success
                    )
                    
                    self.metrics.append(metrics)
                    self._check_performance_thresholds(metrics)
                
                return result
            return wrapper
        return decorator
    
    def _get_token_count(self) -> int:
        """Get current token usage (mock implementation)."""
        return random.randint(0, 1000)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (mock implementation)."""
        return random.uniform(50, 300)
    
    def _get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics (mock implementation)."""
        return {"hits": random.randint(0, 10), "misses": random.randint(0, 5)}
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds."""
        issues = []
        
        if metrics.duration_seconds > self.performance_thresholds["max_duration_seconds"]:
            issues.append(f"Duration exceeded threshold: {metrics.duration_seconds:.2f}s")
        
        if metrics.memory_usage_mb > self.performance_thresholds["max_memory_mb"]:
            issues.append(f"Memory usage exceeded threshold: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.tokens_used > self.performance_thresholds["max_tokens_per_request"]:
            issues.append(f"Token usage exceeded threshold: {metrics.tokens_used}")
        
        if issues:
            logging.warning(f"Performance issues in {metrics.operation_name}: {'; '.join(issues)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {"message": "No performance data available"}
        
        # Calculate statistics
        durations = [m.duration_seconds for m in self.metrics if m.success]
        memory_usage = [m.memory_usage_mb for m in self.metrics]
        token_usage = [m.tokens_used for m in self.metrics]
        
        cache_hits = sum(m.cache_hits for m in self.metrics)
        cache_misses = sum(m.cache_misses for m in self.metrics)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics)
        
        return {
            "total_operations": len(self.metrics),
            "success_rate": success_rate,
            "average_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "average_memory_mb": sum(memory_usage) / len(memory_usage),
            "total_tokens_used": sum(token_usage),
            "cache_hit_rate": cache_hit_rate,
            "operations_per_hour": len([m for m in self.metrics if m.start_time > datetime.now() - timedelta(hours=1)]),
            "optimization_opportunities": self._identify_optimization_opportunities()
        }
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify potential optimization opportunities."""
        opportunities = []
        
        if not self.metrics:
            return opportunities
        
        # Check cache hit rate
        total_hits = sum(m.cache_hits for m in self.metrics)
        total_misses = sum(m.cache_misses for m in self.metrics)
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        if hit_rate < self.performance_thresholds["min_cache_hit_rate"]:
            opportunities.append("Improve caching strategy - low cache hit rate detected")
        
        # Check for slow operations
        slow_operations = [m for m in self.metrics if m.duration_seconds > 10.0]
        if len(slow_operations) > len(self.metrics) * 0.2:  # More than 20% slow
            opportunities.append("Optimize slow operations - consider parallel processing or streamlining")
        
        # Check token efficiency
        avg_tokens = sum(m.tokens_used for m in self.metrics) / len(self.metrics)
        if avg_tokens > 2000:
            opportunities.append("Optimize token usage - consider prompt engineering or response compression")
        
        return opportunities


# =============================================================================
# 3. Debugging and Troubleshooting Tools
# =============================================================================

class AgentDebugger:
    """
    Comprehensive debugging toolkit for AI agents.
    Provides tracing, state inspection, and interactive debugging capabilities.
    """
    
    def __init__(self):
        self.debug_enabled = True
        self.trace_history: List[Dict[str, Any]] = []
        self.breakpoints: Dict[str, bool] = {}
        self.watch_variables: Dict[str, Any] = {}
    
    def trace_execution(self, step_name: str, state: Dict[str, Any], 
                       additional_info: Dict[str, Any] = None):
        """
        Trace agent execution step-by-step.
        
        Args:
            step_name: Name of the execution step
            state: Current agent state
            additional_info: Additional debugging information
        """
        if not self.debug_enabled:
            return
        
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "state_snapshot": self._sanitize_state(state),
            "additional_info": additional_info or {},
            "memory_usage": self._get_current_memory(),
            "step_number": len(self.trace_history) + 1
        }
        
        self.trace_history.append(trace_entry)
        
        # Check for breakpoints
        if step_name in self.breakpoints and self.breakpoints[step_name]:
            self._handle_breakpoint(step_name, trace_entry)
        
        # Log trace entry
        logging.debug(f"TRACE [{trace_entry['step_number']}] {step_name}: {additional_info}")
    
    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for logging (remove sensitive data)."""
        sanitized = {}
        sensitive_keys = ["api_key", "password", "token", "secret"]
        
        for key, value in state.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_state(value)
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:100] + "...[TRUNCATED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_current_memory(self) -> str:
        """Get current memory usage (mock implementation)."""
        return f"{random.randint(100, 500)}MB"
    
    def set_breakpoint(self, step_name: str, enabled: bool = True):
        """Set or remove a breakpoint at a specific step."""
        self.breakpoints[step_name] = enabled
        logging.info(f"Breakpoint {'set' if enabled else 'removed'} for step: {step_name}")
    
    def _handle_breakpoint(self, step_name: str, trace_entry: Dict[str, Any]):
        """Handle breakpoint activation."""
        print(f"\n🛑 BREAKPOINT HIT: {step_name}")
        print(f"Step number: {trace_entry['step_number']}")
        print(f"Timestamp: {trace_entry['timestamp']}")
        print(f"State keys: {list(trace_entry['state_snapshot'].keys())}")
        print(f"Memory usage: {trace_entry['memory_usage']}")
        
        # Interactive debugging prompt (simplified)
        while True:
            command = input("Debug> ").strip().lower()
            
            if command == "continue" or command == "c":
                break
            elif command == "state" or command == "s":
                print(json.dumps(trace_entry['state_snapshot'], indent=2))
            elif command == "history" or command == "h":
                self._show_trace_history(5)  # Last 5 steps
            elif command == "help":
                print("Commands: continue(c), state(s), history(h), help, quit")
            elif command == "quit" or command == "q":
                self.debug_enabled = False
                break
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    def _show_trace_history(self, last_n: int = 10):
        """Show recent trace history."""
        recent_traces = self.trace_history[-last_n:]
        
        print(f"\n📊 Last {len(recent_traces)} execution steps:")
        for trace in recent_traces:
            print(f"  {trace['step_number']}: {trace['step_name']} ({trace['timestamp']})")
    
    def analyze_execution_patterns(self) -> Dict[str, Any]:
        """Analyze execution patterns for optimization insights."""
        if not self.trace_history:
            return {"message": "No execution trace available"}
        
        # Analyze step frequency
        step_counts = {}
        step_durations = {}
        
        for i, trace in enumerate(self.trace_history):
            step_name = trace['step_name']
            step_counts[step_name] = step_counts.get(step_name, 0) + 1
            
            # Calculate duration (simplified)
            if i > 0:
                prev_time = datetime.fromisoformat(self.trace_history[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(trace['timestamp'])
                duration = (curr_time - prev_time).total_seconds()
                
                if step_name not in step_durations:
                    step_durations[step_name] = []
                step_durations[step_name].append(duration)
        
        # Calculate average durations
        avg_durations = {}
        for step, durations in step_durations.items():
            avg_durations[step] = sum(durations) / len(durations)
        
        return {
            "total_steps_traced": len(self.trace_history),
            "unique_steps": len(step_counts),
            "step_frequency": step_counts,
            "average_step_durations": avg_durations,
            "slowest_steps": sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_frequent_steps": sorted(step_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


# =============================================================================
# 4. Production Deployment Best Practices
# =============================================================================

class ProductionAgentManager:
    """
    Production-ready agent management system with health checks,
    scaling, and deployment best practices.
    """
    
    def __init__(self):
        self.health_checks = {}
        self.scaling_config = {}
        self.deployment_config = {}
        self.setup_production_monitoring()
    
    def setup_production_monitoring(self):
        """Setup monitoring for production deployment."""
        self.health_checks = {
            "llm_connectivity": self._check_llm_connectivity,
            "memory_usage": self._check_memory_usage,
            "response_times": self._check_response_times,
            "error_rates": self._check_error_rates,
            "api_quotas": self._check_api_quotas
        }
        
        self.scaling_config = {
            "auto_scaling_enabled": True,
            "min_instances": 1,
            "max_instances": 10,
            "cpu_threshold": 70,
            "memory_threshold": 80,
            "request_threshold": 100  # requests per minute
        }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {}
        overall_health = True
        
        for check_name, check_function in self.health_checks.items():
            try:
                result = check_function()
                health_status[check_name] = result
                
                if not result.get("healthy", False):
                    overall_health = False
                    
            except Exception as e:
                health_status[check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "checked_at": datetime.now().isoformat()
                }
                overall_health = False
        
        return {
            "overall_health": overall_health,
            "status": "healthy" if overall_health else "unhealthy",
            "checks": health_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_llm_connectivity(self) -> Dict[str, Any]:
        """Check LLM service connectivity."""
        # Mock connectivity check
        return {
            "healthy": True,
            "response_time_ms": random.randint(100, 500),
            "endpoint": "api.openai.com",
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage levels."""
        memory_usage = random.uniform(40, 90)  # Mock percentage
        
        return {
            "healthy": memory_usage < 85,
            "memory_usage_percent": memory_usage,
            "threshold": 85,
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_response_times(self) -> Dict[str, Any]:
        """Check average response times."""
        avg_response_time = random.uniform(0.5, 5.0)  # Mock seconds
        
        return {
            "healthy": avg_response_time < 3.0,
            "avg_response_time_seconds": avg_response_time,
            "threshold_seconds": 3.0,
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_error_rates(self) -> Dict[str, Any]:
        """Check error rates."""
        error_rate = random.uniform(0, 10)  # Mock percentage
        
        return {
            "healthy": error_rate < 5,
            "error_rate_percent": error_rate,
            "threshold_percent": 5,
            "checked_at": datetime.now().isoformat()
        }
    
    def _check_api_quotas(self) -> Dict[str, Any]:
        """Check API quota usage."""
        quota_usage = random.uniform(60, 95)  # Mock percentage
        
        return {
            "healthy": quota_usage < 90,
            "quota_usage_percent": quota_usage,
            "threshold_percent": 90,
            "checked_at": datetime.now().isoformat()
        }
    
    def get_deployment_recommendations(self) -> List[str]:
        """Get deployment recommendations based on best practices."""
        recommendations = [
            "Implement circuit breakers for external API calls",
            "Use connection pooling for database connections",
            "Enable distributed tracing for complex workflows",
            "Implement rate limiting to prevent abuse",
            "Use blue-green deployment for zero-downtime updates",
            "Monitor and alert on key performance metrics",
            "Implement graceful shutdown procedures",
            "Use secrets management for API keys",
            "Enable comprehensive logging and audit trails",
            "Implement request/response validation",
            "Use load balancing for high availability",
            "Enable automatic scaling based on demand"
        ]
        
        return recommendations


# =============================================================================
# 5. Common Issues and Solutions
# =============================================================================

class TroubleshootingGuide:
    """
    Comprehensive troubleshooting guide for common AI agent issues.
    """
    
    def __init__(self):
        self.issue_database = self._load_issue_database()
    
    def _load_issue_database(self) -> Dict[str, Dict[str, Any]]:
        """Load database of common issues and solutions."""
        return {
            "tool_not_called": {
                "description": "Agent is not calling the expected tool",
                "common_causes": [
                    "Tool description is unclear or incomplete",
                    "Parameter types don't match expectations",
                    "Tool name is not descriptive enough"
                ],
                "solutions": [
                    "Improve tool description with clear examples",
                    "Add explicit parameter type hints",
                    "Use descriptive tool names that match user intent",
                    "Test tool descriptions with different phrasings"
                ],
                "severity": "medium"
            },
            "json_parsing_errors": {
                "description": "Agent output cannot be parsed as JSON",
                "common_causes": [
                    "LLM generating malformed JSON",
                    "Response includes extra text outside JSON",
                    "Special characters breaking JSON format"
                ],
                "solutions": [
                    "Switch to structured-chat agent",
                    "Add JSON validation in tool responses",
                    "Use output parsers to clean LLM responses",
                    "Implement retry logic with format correction"
                ],
                "severity": "high"
            },
            "high_latency": {
                "description": "Agent responses are too slow",
                "common_causes": [
                    "No caching of tool outputs",
                    "Sequential tool calls instead of parallel",
                    "Large context windows or verbose prompts"
                ],
                "solutions": [
                    "Implement tool output caching",
                    "Use parallel tool execution where possible",
                    "Optimize prompt length and complexity",
                    "Consider streaming responses for long operations"
                ],
                "severity": "medium"
            },
            "memory_issues": {
                "description": "Agent consuming too much memory",
                "common_causes": [
                    "Large conversation histories",
                    "Caching too much data",
                    "Memory leaks in tool implementations"
                ],
                "solutions": [
                    "Implement conversation summarization",
                    "Use LRU cache with size limits",
                    "Profile tool implementations for memory leaks",
                    "Clear intermediate results after processing"
                ],
                "severity": "high"
            },
            "authentication_failures": {
                "description": "API authentication errors",
                "common_causes": [
                    "Invalid or expired API keys",
                    "Incorrect API endpoint configuration",
                    "Rate limiting or quota exceeded"
                ],
                "solutions": [
                    "Validate API keys and refresh if needed",
                    "Implement proper error handling for auth failures",
                    "Add retry logic with exponential backoff",
                    "Monitor API usage and quotas"
                ],
                "severity": "critical"
            }
        }
    
    def diagnose_issue(self, symptoms: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Diagnose issues based on symptoms and provide solutions.
        
        Args:
            symptoms: List of observed symptoms
            context: Additional context about the issue
            
        Returns:
            Dictionary with diagnosis and recommended solutions
        """
        context = context or {}
        matched_issues = []
        
        # Match symptoms to known issues
        for issue_key, issue_data in self.issue_database.items():
            relevance_score = 0
            
            # Check if any symptoms match the issue description or causes
            issue_text = (issue_data["description"] + " " + 
                         " ".join(issue_data["common_causes"])).lower()
            
            for symptom in symptoms:
                if symptom.lower() in issue_text:
                    relevance_score += 1
            
            if relevance_score > 0:
                matched_issues.append({
                    "issue": issue_key,
                    "relevance_score": relevance_score,
                    "data": issue_data
                })
        
        # Sort by relevance
        matched_issues.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not matched_issues:
            return {
                "diagnosis": "No matching issues found",
                "recommendations": [
                    "Check agent logs for specific error messages",
                    "Review recent changes to agent configuration",
                    "Test individual tools in isolation",
                    "Verify environment and dependencies"
                ]
            }
        
        # Return top match with solutions
        top_issue = matched_issues[0]
        
        return {
            "diagnosis": top_issue["data"]["description"],
            "severity": top_issue["data"]["severity"],
            "likely_causes": top_issue["data"]["common_causes"],
            "recommended_solutions": top_issue["data"]["solutions"],
            "alternative_issues": [issue["issue"] for issue in matched_issues[1:3]],
            "context_analysis": self._analyze_context(context),
            "diagnosis_confidence": min(top_issue["relevance_score"] / len(symptoms), 1.0)
        }
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Analyze context for additional insights."""
        analysis = {}
        
        if context.get("error_rate", 0) > 0.1:
            analysis["error_rate"] = "High error rate detected - check for systematic issues"
        
        if context.get("response_time", 0) > 5:
            analysis["performance"] = "Slow response times - consider optimization"
        
        if context.get("memory_usage", 0) > 80:
            analysis["memory"] = "High memory usage - check for memory leaks"
        
        return analysis


# =============================================================================
# 6. Demonstration Functions
# =============================================================================

def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("🚨 Error Handling Demonstration")
    print("=" * 50)
    
    error_handler = AgentErrorHandler()
    
    # Simulate different types of errors
    test_errors = [
        (ValueError("Invalid input format"), "input_validator"),
        (ConnectionError("API endpoint unreachable"), "llm_interface"),
        (KeyError("Missing required field"), "tool_executor"),
        (TimeoutError("Request timed out"), "external_api")
    ]
    
    print("\n🔧 Simulating various error scenarios:")
    
    for error, component in test_errors:
        context = {"operation": "test_operation", "timestamp": datetime.now().isoformat()}
        agent_error = error_handler.handle_error(error, component, context)
        
        print(f"\n❌ {agent_error.error_id}: {agent_error.message}")
        print(f"   Severity: {agent_error.severity.value}")
        print(f"   Component: {agent_error.component}")
        print(f"   Suggestions: {len(agent_error.resolution_suggestions)}")
    
    # Show error statistics
    print(f"\n📊 Error Statistics:")
    stats = error_handler.get_error_statistics()
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Most problematic: {stats['most_problematic_component']}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n⚡ Performance Monitoring Demonstration")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Simulate monitored operations
    @monitor.monitor_operation("test_calculation")
    def slow_calculation():
        time.sleep(random.uniform(0.1, 0.5))  # Simulate work
        return "calculation_result"
    
    @monitor.monitor_operation("api_call")
    def mock_api_call():
        time.sleep(random.uniform(0.05, 0.2))  # Simulate API call
        return {"status": "success", "data": "mock_data"}
    
    print("\n⏱️  Running monitored operations:")
    
    # Execute operations
    for i in range(5):
        slow_calculation()
        mock_api_call()
        print(f"   Completed operation set {i+1}")
    
    # Show performance report
    print(f"\n📈 Performance Report:")
    report = monitor.get_performance_report()
    print(f"   Total operations: {report['total_operations']}")
    print(f"   Success rate: {report['success_rate']:.1%}")
    print(f"   Average duration: {report['average_duration_seconds']:.3f}s")
    print(f"   Cache hit rate: {report['cache_hit_rate']:.1%}")
    
    if report['optimization_opportunities']:
        print(f"\n💡 Optimization Opportunities:")
        for opportunity in report['optimization_opportunities']:
            print(f"   • {opportunity}")


def demonstrate_debugging_tools():
    """Demonstrate debugging and tracing capabilities."""
    print("\n🐛 Debugging Tools Demonstration")
    print("=" * 50)
    
    debugger = AgentDebugger()
    
    # Simulate agent execution with tracing
    def simulate_agent_execution():
        state = {"step": 0, "results": []}
        
        debugger.trace_execution("initialization", state, {"info": "Starting agent"})
        
        state["step"] = 1
        state["results"].append("tool_1_result")
        debugger.trace_execution("tool_execution", state, {"tool": "calculator"})
        
        state["step"] = 2  
        state["results"].append("tool_2_result")
        debugger.trace_execution("tool_execution", state, {"tool": "search"})
        
        state["step"] = 3
        state["final_answer"] = "Complete"
        debugger.trace_execution("completion", state, {"status": "success"})
    
    print("\n🔍 Tracing agent execution:")
    simulate_agent_execution()
    
    # Show execution analysis
    print(f"\n📊 Execution Analysis:")
    analysis = debugger.analyze_execution_patterns()
    print(f"   Total steps traced: {analysis['total_steps_traced']}")
    print(f"   Unique step types: {analysis['unique_steps']}")
    
    if analysis['most_frequent_steps']:
        print(f"   Most frequent steps:")
        for step, count in analysis['most_frequent_steps'][:3]:
            print(f"     • {step}: {count} times")


def demonstrate_troubleshooting():
    """Demonstrate troubleshooting guide capabilities."""
    print("\n🔧 Troubleshooting Guide Demonstration")
    print("=" * 50)
    
    guide = TroubleshootingGuide()
    
    # Test different issue scenarios
    test_scenarios = [
        {
            "symptoms": ["tool not being called", "agent ignoring requests"],
            "context": {"error_rate": 0.3}
        },
        {
            "symptoms": ["slow response", "high latency", "timeout"],
            "context": {"response_time": 8.5, "memory_usage": 45}
        },
        {
            "symptoms": ["json error", "parsing failed", "malformed output"],
            "context": {"error_rate": 0.15}
        }
    ]
    
    print("\n🩺 Diagnosing common issues:")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Symptoms: {', '.join(scenario['symptoms'])}")
        
        diagnosis = guide.diagnose_issue(scenario["symptoms"], scenario["context"])
        
        print(f"   🔍 Diagnosis: {diagnosis['diagnosis']}")
        print(f"   ⚠️  Severity: {diagnosis.get('severity', 'unknown')}")
        print(f"   💡 Top solution: {diagnosis['recommended_solutions'][0]}")
        print(f"   🎯 Confidence: {diagnosis['diagnosis_confidence']:.1%}")


# Main demonstration
if __name__ == "__main__":
    print("🎓 Module 7: Best Practices & Troubleshooting")
    print("=" * 60)
    
    # Import random for mock implementations
    import random
    
    # Demonstrate all best practices
    demonstrate_error_handling()
    demonstrate_performance_monitoring() 
    demonstrate_debugging_tools()
    demonstrate_troubleshooting()
    
    print("\n✅ Module 7 demonstrations completed!")
    print("\n🎉 COURSE COMPLETE! 🎉")
    print("=" * 60)
    print("You now have a comprehensive understanding of:")
    print("  • AI Agent fundamentals and architectures")
    print("  • Custom tool development and integration")
    print("  • LangGraph orchestration and workflows")
    print("  • Real-world application development")
    print("  • Production best practices and troubleshooting")
    print("\n🚀 Ready to build production-grade AI agents!"))
