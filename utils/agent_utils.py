"""
Utility functions for AI Agents development
==========================================

Common utilities for agent development, debugging, monitoring, and testing.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from functools import wraps


# =============================================================================
# Agent Performance Monitoring
# =============================================================================

@dataclass
class AgentExecutionMetrics:
    """Metrics for tracking agent execution performance."""
    start_time: float
    end_time: Optional[float] = None
    total_steps: int = 0
    tools_used: List[str] = None
    errors_encountered: List[str] = None
    memory_usage: Optional[float] = None
    success: bool = False
    
    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []
        if self.errors_encountered is None:
            self.errors_encountered = []
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        data['duration'] = self.duration
        return data


class AgentMonitor:
    """Monitor and track agent performance metrics."""
    
    def __init__(self, log_level: str = "INFO"):
        self.execution_history: List[AgentExecutionMetrics] = []
        self.current_execution: Optional[AgentExecutionMetrics] = None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def start_execution(self) -> AgentExecutionMetrics:
        """Start monitoring a new agent execution."""
        self.current_execution = AgentExecutionMetrics(start_time=time.time())
        self.logger.info("Started monitoring agent execution")
        return self.current_execution
    
    def log_tool_usage(self, tool_name: str):
        """Log tool usage during execution."""
        if self.current_execution:
            self.current_execution.tools_used.append(tool_name)
            self.logger.debug(f"Tool used: {tool_name}")
    
    def log_step(self):
        """Log a step in agent execution."""
        if self.current_execution:
            self.current_execution.total_steps += 1
            self.logger.debug(f"Step {self.current_execution.total_steps} completed")
    
    def log_error(self, error: str):
        """Log an error during execution."""
        if self.current_execution:
            self.current_execution.errors_encountered.append(error)
            self.logger.error(f"Agent error: {error}")
    
    def end_execution(self, success: bool = True) -> AgentExecutionMetrics:
        """End monitoring and store results."""
        if self.current_execution:
            self.current_execution.end_time = time.time()
            self.current_execution.success = success
            
            # Store in history
            self.execution_history.append(self.current_execution)
            
            self.logger.info(
                f"Execution completed - Duration: {self.current_execution.duration:.2f}s, "
                f"Steps: {self.current_execution.total_steps}, "
                f"Success: {success}"
            )
            
            completed_execution = self.current_execution
            self.current_execution = None
            return completed_execution
        
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored executions."""
        if not self.execution_history:
            return {"message": "No executions recorded"}
        
        durations = [ex.duration for ex in self.execution_history if ex.duration]
        success_rate = sum(1 for ex in self.execution_history if ex.success) / len(self.execution_history)
        
        all_tools = []
        all_errors = []
        total_steps = 0
        
        for execution in self.execution_history:
            all_tools.extend(execution.tools_used)
            all_errors.extend(execution.errors_encountered)
            total_steps += execution.total_steps
        
        tool_usage = {}
        for tool in all_tools:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return {
            "total_executions": len(self.execution_history),
            "success_rate": round(success_rate * 100, 2),
            "average_duration": round(sum(durations) / len(durations), 2) if durations else 0,
            "average_steps": round(total_steps / len(self.execution_history), 2),
            "most_used_tools": sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            "total_errors": len(all_errors),
            "common_errors": list(set(all_errors))[:5]
        }


def monitor_agent_execution(monitor: AgentMonitor):
    """Decorator to automatically monitor agent function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = monitor.start_execution()
            try:
                result = func(*args, **kwargs)
                monitor.end_execution(success=True)
                return result
            except Exception as e:
                monitor.log_error(str(e))
                monitor.end_execution(success=False)
                raise
        return wrapper
    return decorator


# =============================================================================
# Agent Debugging Utilities
# =============================================================================

class AgentDebugger:
    """Debugging utilities for AI agents."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.debug_log: List[Dict[str, Any]] = []
    
    def log_thought_process(self, step: str, content: str, metadata: Dict[str, Any] = None):
        """Log agent's thought process for debugging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "content": content,
            "metadata": metadata or {}
        }
        
        self.debug_log.append(log_entry)
        
        if self.verbose:
            print(f"🧠 [{step}]: {content}")
            if metadata:
                print(f"   Metadata: {metadata}")
    
    def log_tool_call(self, tool_name: str, inputs: Dict[str, Any], outputs: Any):
        """Log tool calls for debugging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool": tool_name,
            "inputs": inputs,
            "outputs": outputs
        }
        
        self.debug_log.append(log_entry)
        
        if self.verbose:
            print(f"🔧 Tool Call: {tool_name}")
            print(f"   Inputs: {inputs}")
            print(f"   Outputs: {str(outputs)[:100]}..." if len(str(outputs)) > 100 else f"   Outputs: {outputs}")
    
    def log_decision(self, decision: str, reasoning: str, confidence: float = None):
        """Log agent decisions and reasoning."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "decision",
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence
        }
        
        self.debug_log.append(log_entry)
        
        if self.verbose:
            print(f"⚡ Decision: {decision}")
            print(f"   Reasoning: {reasoning}")
            if confidence:
                print(f"   Confidence: {confidence:.2f}")
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get complete execution trace."""
        return self.debug_log.copy()
    
    def export_trace(self, filename: str):
        """Export execution trace to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        print(f"Debug trace exported to {filename}")
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze execution for performance bottlenecks."""
        tool_calls = [entry for entry in self.debug_log if entry.get("type") == "tool_call"]
        
        if not tool_calls:
            return {"message": "No tool calls to analyze"}
        
        # Tool usage frequency
        tool_frequency = {}
        for call in tool_calls:
            tool = call["tool"]
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
        
        # Find potential inefficiencies
        issues = []
        if len(tool_calls) > 20:
            issues.append("High number of tool calls - consider optimizing agent logic")
        
        # Check for repeated identical calls
        call_signatures = []
        for call in tool_calls:
            signature = f"{call['tool']}:{json.dumps(call['inputs'], sort_keys=True)}"
            call_signatures.append(signature)
        
        duplicate_calls = len(call_signatures) - len(set(call_signatures))
        if duplicate_calls > 0:
            issues.append(f"Found {duplicate_calls} duplicate tool calls - consider caching")
        
        return {
            "total_tool_calls": len(tool_calls),
            "unique_tools_used": len(tool_frequency),
            "tool_frequency": tool_frequency,
            "duplicate_calls": duplicate_calls,
            "potential_issues": issues
        }


# =============================================================================
# Agent Testing Framework
# =============================================================================

class AgentTestCase:
    """Test case for agent functionality."""
    
    def __init__(self, name: str, input_data: Any, expected_behavior: str, validation_func: Callable = None):
        self.name = name
        self.input_data = input_data
        self.expected_behavior = expected_behavior
        self.validation_func = validation_func
        self.result = None
        self.passed = None
        self.execution_time = None
        self.error = None


class AgentTester:
    """Framework for testing agent functionality."""
    
    def __init__(self):
        self.test_cases: List[AgentTestCase] = []
        self.results: List[Dict[str, Any]] = []
    
    def add_test_case(self, name: str, input_data: Any, expected_behavior: str, validation_func: Callable = None):
        """Add a test case."""
        test_case = AgentTestCase(name, input_data, expected_behavior, validation_func)
        self.test_cases.append(test_case)
    
    def run_tests(self, agent_func: Callable) -> Dict[str, Any]:
        """Run all test cases against the agent function."""
        print(f"🧪 Running {len(self.test_cases)} test cases...")
        
        passed = 0
        failed = 0
        
        for test_case in self.test_cases:
            print(f"\n📝 Test: {test_case.name}")
            
            start_time = time.time()
            
            try:
                # Execute agent with test input
                result = agent_func(test_case.input_data)
                test_case.result = result
                test_case.execution_time = time.time() - start_time
                
                # Validate result
                if test_case.validation_func:
                    test_case.passed = test_case.validation_func(result)
                else:
                    # Default validation: check if result exists and no error
                    test_case.passed = result is not None and not (isinstance(result, dict) and "error" in result)
                
                if test_case.passed:
                    print(f"   ✅ PASSED ({test_case.execution_time:.2f}s)")
                    passed += 1
                else:
                    print(f"   ❌ FAILED ({test_case.execution_time:.2f}s)")
                    print(f"      Expected: {test_case.expected_behavior}")
                    print(f"      Got: {str(result)[:100]}...")
                    failed += 1
                    
            except Exception as e:
                test_case.error = str(e)
                test_case.passed = False
                test_case.execution_time = time.time() - start_time
                print(f"   💥 ERROR ({test_case.execution_time:.2f}s): {e}")
                failed += 1
        
        # Generate test report
        total_time = sum(tc.execution_time for tc in self.test_cases if tc.execution_time)
        
        report = {
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(self.test_cases)) * 100 if self.test_cases else 0,
            "total_execution_time": round(total_time, 2),
            "average_execution_time": round(total_time / len(self.test_cases), 2) if self.test_cases else 0
        }
        
        print(f"\n📊 Test Results:")
        print(f"   Total: {report['total_tests']}")
        print(f"   Passed: {report['passed']}")
        print(f"   Failed: {report['failed']}")
        print(f"   Success Rate: {report['success_rate']:.1f}%")
        print(f"   Total Time: {report['total_execution_time']}s")
        
        return report


# =============================================================================
# Configuration Management
# =============================================================================

class AgentConfig:
    """Configuration management for AI agents."""
    
    def __init__(self, config_file: str = None):
        self.config_data = {}
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                self.config_data = json.load(f)
            print(f"✅ Configuration loaded from {config_file}")
        except FileNotFoundError:
            print(f"⚠️  Configuration file {config_file} not found, using defaults")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        with open(config_file, 'w') as f:
            json.dump(self.config_data, f, indent=2)
        print(f"✅ Configuration saved to {config_file}")
    
    def get_default_agent_config(self) -> Dict[str, Any]:
        """Get default agent configuration."""
        return {
            "max_iterations": 10,
            "timeout": 300,
            "verbose": True,
            "error_handling": {
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600
            },
            "logging": {
                "level": "INFO",
                "enable_debug": False
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def format_execution_trace(trace: List[Dict[str, Any]]) -> str:
    """Format execution trace for human readability."""
    formatted_lines = []
    
    for i, entry in enumerate(trace, 1):
        timestamp = entry.get("timestamp", "Unknown")
        entry_type = entry.get("type", entry.get("step", "Unknown"))
        
        if entry_type == "tool_call":
            line = f"{i:2d}. [{timestamp[-8:]}] 🔧 {entry['tool']} -> {str(entry['outputs'])[:50]}..."
        elif entry_type == "decision":
            line = f"{i:2d}. [{timestamp[-8:]}] ⚡ {entry['decision']} (confidence: {entry.get('confidence', 'N/A')})"
        else:
            content = entry.get("content", str(entry))
            line = f"{i:2d}. [{timestamp[-8:]}] 🧠 {content[:70]}..."
        
        formatted_lines.append(line)
    
    return "\n".join(formatted_lines)


def validate_tool_schema(tool_func: Callable) -> Dict[str, Any]:
    """Validate tool function schema and return analysis."""
    import inspect
    
    analysis = {
        "name": tool_func.__name__,
        "has_docstring": bool(tool_func.__doc__),
        "has_type_hints": False,
        "parameter_count": 0,
        "issues": []
    }
    
    # Check function signature
    sig = inspect.signature(tool_func)
    analysis["parameter_count"] = len(sig.parameters)
    
    # Check type hints
    type_hints = []
    for param_name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            type_hints.append(f"{param_name}: {param.annotation}")
    
    analysis["has_type_hints"] = len(type_hints) > 0
    analysis["type_hints"] = type_hints
    
    # Check return type hint
    if sig.return_annotation != inspect.Signature.empty:
        analysis["return_type"] = str(sig.return_annotation)
    
    # Identify potential issues
    if not analysis["has_docstring"]:
        analysis["issues"].append("Missing docstring - agents need clear descriptions")
    
    if not analysis["has_type_hints"]:
        analysis["issues"].append("Missing type hints - improves tool reliability")
    
    if analysis["parameter_count"] == 0:
        analysis["issues"].append("No parameters - tools should accept inputs")
    
    if analysis["parameter_count"] > 5:
        analysis["issues"].append("Many parameters - consider using structured inputs")
    
    return analysis


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration file template."""
    return {
        "agent": {
            "max_iterations": 10,
            "timeout": 300,
            "verbose": True
        },
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "max_tokens": 1000
        },
        "tools": {
            "enable_caching": True,
            "cache_ttl": 3600,
            "timeout": 30
        },
        "monitoring": {
            "enable_logging": True,
            "log_level": "INFO",
            "export_traces": False
        },
        "api_keys": {
            "openai": "your_openai_key_here",
            "anthropic": "your_anthropic_key_here"
        }
    }


# Example usage
if __name__ == "__main__":
    print("🛠️  Agent Utilities Demo")
    print("=" * 40)
    
    # Demonstrate monitoring
    monitor = AgentMonitor()
    
    @monitor_agent_execution(monitor)
    def sample_agent_function(query: str):
        """Sample agent function for testing."""
        monitor.log_tool_usage("sample_tool")
        monitor.log_step()
        time.sleep(0.1)  # Simulate processing time
        return f"Processed: {query}"
    
    # Test monitoring
    result = sample_agent_function("test query")
    print(f"Result: {result}")
    
    # Show performance summary
    summary = monitor.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")
    
    print("\n✅ Utilities demonstration completed!")
