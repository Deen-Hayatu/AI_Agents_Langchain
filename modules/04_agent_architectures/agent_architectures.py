"""
Module 4: Agent Architectures
=============================

This module explores different agent architectures including ReAct, Structured Chat,
and Plan-and-Execute patterns. Learn when to use each architecture and how to implement them.

Learning Objectives:
- Understand ReAct (Reason + Act) pattern
- Implement structured chat agents
- Compare different agent architectures
- Learn planning and execution strategies
- Handle complex multi-step workflows
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# =============================================================================
# 1. Agent Architecture Types
# =============================================================================

class AgentType(Enum):
    """Enumeration of different agent architecture types."""
    REACT = "react"
    STRUCTURED_CHAT = "structured_chat"
    PLAN_AND_EXECUTE = "plan_and_execute"
    CONVERSATIONAL_REACT = "conversational_react"
    SELF_ASK = "self_ask"


@dataclass
class AgentStep:
    """Represents a single step in agent execution."""
    step_number: int
    action_type: str  # "thought", "action", "observation", "answer"
    content: str
    tool_used: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[Any] = None
    timestamp: Optional[str] = None


class AgentExecutionTrace:
    """Tracks the complete execution trace of an agent."""
    
    def __init__(self, agent_type: AgentType, task: str):
        self.agent_type = agent_type
        self.task = task
        self.steps: List[AgentStep] = []
        self.start_time = datetime.now()
        self.end_time = None
        self.final_answer = None
    
    def add_step(self, action_type: str, content: str, tool_used: str = None, 
                tool_input: Dict = None, tool_output: Any = None):
        """Add a step to the execution trace."""
        step = AgentStep(
            step_number=len(self.steps) + 1,
            action_type=action_type,
            content=content,
            tool_used=tool_used,
            tool_input=tool_input,
            tool_output=tool_output,
            timestamp=datetime.now().isoformat()
        )
        self.steps.append(step)
    
    def complete_execution(self, final_answer: str):
        """Mark the execution as complete."""
        self.final_answer = final_answer
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else None
        
        return {
            "agent_type": self.agent_type.value,
            "task": self.task,
            "total_steps": len(self.steps),
            "tools_used": list(set(step.tool_used for step in self.steps if step.tool_used)),
            "duration_seconds": duration,
            "final_answer": self.final_answer,
            "success": self.final_answer is not None
        }


# =============================================================================
# 2. ReAct Agent Implementation
# =============================================================================

class ReActAgent:
    """
    Implementation of ReAct (Reason + Act) agent pattern.
    
    ReAct agents follow the pattern:
    1. Thought: Reason about what to do next
    2. Action: Take an action using a tool
    3. Observation: Observe the result
    4. Repeat until task is complete
    """
    
    def __init__(self, tools: List[Dict], max_iterations: int = 10):
        self.tools = {tool["name"]: tool for tool in tools}
        self.max_iterations = max_iterations
    
    def execute(self, task: str) -> AgentExecutionTrace:
        """
        Execute a task using ReAct pattern.
        
        Args:
            task: The task to execute
            
        Returns:
            Complete execution trace
        """
        trace = AgentExecutionTrace(AgentType.REACT, task)
        
        # Initial thought
        trace.add_step("thought", f"I need to solve: {task}")
        
        for iteration in range(self.max_iterations):
            # Determine next action
            action_plan = self._plan_next_action(task, trace.steps)
            
            if action_plan["action"] == "final_answer":
                trace.add_step("answer", action_plan["content"])
                trace.complete_execution(action_plan["content"])
                break
            
            # Add thought step
            trace.add_step("thought", action_plan["reasoning"])
            
            # Execute action
            if action_plan["action"] in self.tools:
                tool_result = self._execute_tool(
                    action_plan["action"], 
                    action_plan["input"]
                )
                
                trace.add_step(
                    "action",
                    f"Using {action_plan['action']} with input: {action_plan['input']}",
                    tool_used=action_plan["action"],
                    tool_input=action_plan["input"],
                    tool_output=tool_result
                )
                
                # Observation
                trace.add_step("observation", f"Result: {tool_result}")
            else:
                trace.add_step("observation", "Tool not available")
        
        if not trace.final_answer:
            trace.complete_execution("Could not complete task within iteration limit")
        
        return trace
    
    def _plan_next_action(self, task: str, steps: List[AgentStep]) -> Dict[str, Any]:
        """Plan the next action based on task and current progress."""
        
        # Simple heuristic-based planning (in real implementation, this would use LLM)
        if "calculate" in task.lower() or "math" in task.lower():
            if not any(step.tool_used == "calculator" for step in steps):
                return {
                    "action": "calculator",
                    "input": {"expression": "2+2"},  # Simplified
                    "reasoning": "I need to perform a calculation"
                }
            else:
                return {
                    "action": "final_answer",
                    "content": "Calculation completed",
                    "reasoning": "I have performed the required calculation"
                }
        
        elif "weather" in task.lower():
            if not any(step.tool_used == "weather" for step in steps):
                return {
                    "action": "weather",
                    "input": {"location": "New York"},
                    "reasoning": "I need to check the weather"
                }
            else:
                return {
                    "action": "final_answer",
                    "content": "Weather information retrieved",
                    "reasoning": "I have obtained the weather information"
                }
        
        elif "search" in task.lower():
            if not any(step.tool_used == "search" for step in steps):
                return {
                    "action": "search",
                    "input": {"query": "information"},
                    "reasoning": "I need to search for information"
                }
            else:
                return {
                    "action": "final_answer",
                    "content": "Search completed",
                    "reasoning": "I have found the requested information"
                }
        
        else:
            return {
                "action": "final_answer",
                "content": f"I understand the task: {task}",
                "reasoning": "Task analysis complete"
            }
    
    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool and return the result."""
        if tool_name in self.tools:
            # Mock tool execution
            tool = self.tools[tool_name]
            if tool_name == "calculator":
                return "42"  # Mock calculation result
            elif tool_name == "weather":
                return {"temperature": 75, "condition": "sunny"}
            elif tool_name == "search":
                return ["Result 1", "Result 2", "Result 3"]
            else:
                return f"Executed {tool_name} with {tool_input}"
        else:
            return f"Tool {tool_name} not found"


# =============================================================================
# 3. Structured Chat Agent Implementation
# =============================================================================

class StructuredChatAgent:
    """
    Implementation of Structured Chat agent for complex inputs/outputs.
    
    Structured Chat agents can handle:
    - Complex JSON inputs/outputs
    - Multi-parameter tool calls
    - Structured reasoning processes
    """
    
    def __init__(self, tools: List[Dict]):
        self.tools = {tool["name"]: tool for tool in tools}
        self.conversation_memory = []
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentExecutionTrace:
        """
        Execute task using structured chat pattern.
        
        Args:
            task: The task to execute
            context: Additional context information
            
        Returns:
            Complete execution trace
        """
        trace = AgentExecutionTrace(AgentType.STRUCTURED_CHAT, task)
        
        # Add context to memory
        if context:
            self.conversation_memory.append({"type": "context", "data": context})
        
        # Structured reasoning
        reasoning_result = self._structured_reasoning(task, context)
        trace.add_step("thought", reasoning_result["analysis"])
        
        # Execute planned actions
        for action in reasoning_result["planned_actions"]:
            trace.add_step("thought", f"Executing: {action['description']}")
            
            if action["tool"] in self.tools:
                result = self._execute_structured_tool(action["tool"], action["parameters"])
                
                trace.add_step(
                    "action",
                    f"Used {action['tool']}",
                    tool_used=action["tool"],
                    tool_input=action["parameters"],
                    tool_output=result
                )
                
                trace.add_step("observation", f"Tool result: {result}")
            else:
                trace.add_step("observation", f"Tool {action['tool']} not available")
        
        # Generate structured response
        final_response = self._generate_structured_response(reasoning_result, trace.steps)
        trace.add_step("answer", json.dumps(final_response, indent=2))
        trace.complete_execution(json.dumps(final_response))
        
        return trace
    
    def _structured_reasoning(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform structured reasoning about the task."""
        
        analysis = {
            "task_type": self._classify_task(task),
            "complexity": self._assess_complexity(task),
            "required_tools": self._identify_required_tools(task),
            "context_relevance": self._assess_context_relevance(context) if context else "No context provided"
        }
        
        planned_actions = []
        
        # Plan actions based on analysis
        if "calculation" in analysis["task_type"]:
            planned_actions.append({
                "tool": "calculator",
                "parameters": {"expression": "sample calculation"},
                "description": "Perform mathematical calculation"
            })
        
        if "research" in analysis["task_type"]:
            planned_actions.append({
                "tool": "search",
                "parameters": {"query": "research topic", "num_results": 5},
                "description": "Research information"
            })
        
        if "analysis" in analysis["task_type"]:
            planned_actions.append({
                "tool": "analyzer",
                "parameters": {"data": "sample data"},
                "description": "Analyze data"
            })
        
        return {
            "analysis": f"Task analysis: {analysis}",
            "planned_actions": planned_actions,
            "execution_strategy": "Sequential execution with validation"
        }
    
    def _classify_task(self, task: str) -> str:
        """Classify the type of task."""
        task_lower = task.lower()
        task_types = []
        
        if any(word in task_lower for word in ["calculate", "compute", "math"]):
            task_types.append("calculation")
        if any(word in task_lower for word in ["search", "find", "research"]):
            task_types.append("research")
        if any(word in task_lower for word in ["analyze", "examine", "evaluate"]):
            task_types.append("analysis")
        if any(word in task_lower for word in ["compare", "contrast", "difference"]):
            task_types.append("comparison")
        
        return "_".join(task_types) if task_types else "general"
    
    def _assess_complexity(self, task: str) -> str:
        """Assess task complexity."""
        word_count = len(task.split())
        question_marks = task.count("?")
        and_or_count = task.lower().count(" and ") + task.lower().count(" or ")
        
        if word_count > 20 or question_marks > 2 or and_or_count > 1:
            return "high"
        elif word_count > 10 or question_marks > 1 or and_or_count > 0:
            return "medium"
        else:
            return "low"
    
    def _identify_required_tools(self, task: str) -> List[str]:
        """Identify which tools are needed for the task."""
        required_tools = []
        task_lower = task.lower()
        
        tool_keywords = {
            "calculator": ["calculate", "compute", "math", "add", "multiply"],
            "search": ["search", "find", "research", "look up"],
            "weather": ["weather", "temperature", "forecast"],
            "analyzer": ["analyze", "examine", "evaluate", "assess"]
        }
        
        for tool, keywords in tool_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                required_tools.append(tool)
        
        return required_tools
    
    def _assess_context_relevance(self, context: Dict[str, Any]) -> str:
        """Assess how relevant the provided context is."""
        if not context:
            return "No context"
        
        relevance_score = 0
        if "user_preferences" in context:
            relevance_score += 2
        if "historical_data" in context:
            relevance_score += 2
        if "constraints" in context:
            relevance_score += 1
        
        if relevance_score >= 4:
            return "Highly relevant"
        elif relevance_score >= 2:
            return "Moderately relevant"
        else:
            return "Low relevance"
    
    def _execute_structured_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """Execute tool with structured input/output."""
        # Mock structured tool execution
        base_result = {
            "tool": tool_name,
            "parameters": parameters,
            "execution_time": datetime.now().isoformat(),
            "success": True
        }
        
        if tool_name == "calculator":
            base_result["result"] = {"calculation": "42", "expression": parameters.get("expression")}
        elif tool_name == "search":
            base_result["result"] = {
                "results": ["Result 1", "Result 2"],
                "total_found": 2,
                "query": parameters.get("query")
            }
        elif tool_name == "analyzer":
            base_result["result"] = {
                "analysis": "Data analyzed successfully",
                "insights": ["Insight 1", "Insight 2"],
                "confidence": 0.85
            }
        else:
            base_result["result"] = f"Executed {tool_name}"
        
        return base_result
    
    def _generate_structured_response(self, reasoning: Dict, steps: List[AgentStep]) -> Dict[str, Any]:
        """Generate a structured response."""
        return {
            "task_completion": {
                "status": "completed",
                "reasoning_used": reasoning["analysis"],
                "actions_taken": len([s for s in steps if s.action_type == "action"]),
                "tools_used": list(set(s.tool_used for s in steps if s.tool_used))
            },
            "results": {
                "primary_outcome": "Task completed successfully",
                "supporting_data": [s.tool_output for s in steps if s.tool_output],
                "confidence_level": 0.9
            },
            "metadata": {
                "execution_path": reasoning["planned_actions"],
                "total_steps": len(steps),
                "completion_time": datetime.now().isoformat()
            }
        }


# =============================================================================
# 4. Plan-and-Execute Agent Implementation
# =============================================================================

class PlanAndExecuteAgent:
    """
    Implementation of Plan-and-Execute agent pattern.
    
    This agent:
    1. Creates a comprehensive plan
    2. Executes the plan step by step
    3. Adapts the plan if needed
    4. Validates results at each step
    """
    
    def __init__(self, tools: List[Dict]):
        self.tools = {tool["name"]: tool for tool in tools}
        self.plan = None
        self.execution_state = {}
    
    def execute(self, task: str) -> AgentExecutionTrace:
        """Execute task using plan-and-execute pattern."""
        trace = AgentExecutionTrace(AgentType.PLAN_AND_EXECUTE, task)
        
        # Phase 1: Planning
        trace.add_step("thought", "Creating execution plan...")
        self.plan = self._create_plan(task)
        trace.add_step("thought", f"Plan created with {len(self.plan['steps'])} steps")
        
        # Phase 2: Execution
        for i, plan_step in enumerate(self.plan["steps"]):
            trace.add_step("thought", f"Executing step {i+1}: {plan_step['description']}")
            
            # Execute step
            step_result = self._execute_plan_step(plan_step)
            
            trace.add_step(
                "action",
                plan_step["description"],
                tool_used=plan_step.get("tool"),
                tool_input=plan_step.get("input"),
                tool_output=step_result
            )
            
            # Validate step result
            validation = self._validate_step_result(plan_step, step_result)
            trace.add_step("observation", f"Step validation: {validation['status']}")
            
            # Store state
            self.execution_state[f"step_{i+1}"] = {
                "result": step_result,
                "validation": validation
            }
            
            # Adapt plan if needed
            if not validation["success"]:
                trace.add_step("thought", "Adapting plan due to step failure...")
                self._adapt_plan(i, validation)
        
        # Phase 3: Final synthesis
        final_result = self._synthesize_results()
        trace.add_step("answer", final_result)
        trace.complete_execution(final_result)
        
        return trace
    
    def _create_plan(self, task: str) -> Dict[str, Any]:
        """Create a comprehensive execution plan."""
        task_lower = task.lower()
        
        # Analyze task complexity and requirements
        plan_steps = []
        
        if "calculate" in task_lower and "research" in task_lower:
            # Complex multi-step task
            plan_steps = [
                {
                    "step_id": 1,
                    "description": "Research relevant information",
                    "tool": "search",
                    "input": {"query": "research topic"},
                    "expected_output": "Information gathered",
                    "dependencies": []
                },
                {
                    "step_id": 2,
                    "description": "Extract numerical data from research",
                    "tool": "analyzer",
                    "input": {"data": "research_results"},
                    "expected_output": "Numerical data extracted",
                    "dependencies": [1]
                },
                {
                    "step_id": 3,
                    "description": "Perform calculations on extracted data",
                    "tool": "calculator",
                    "input": {"expression": "data_calculation"},
                    "expected_output": "Calculation results",
                    "dependencies": [2]
                },
                {
                    "step_id": 4,
                    "description": "Synthesize final answer",
                    "tool": None,
                    "input": None,
                    "expected_output": "Complete answer",
                    "dependencies": [1, 2, 3]
                }
            ]
        elif "calculate" in task_lower:
            plan_steps = [
                {
                    "step_id": 1,
                    "description": "Perform calculation",
                    "tool": "calculator",
                    "input": {"expression": "calculation"},
                    "expected_output": "Calculation result",
                    "dependencies": []
                }
            ]
        elif "research" in task_lower:
            plan_steps = [
                {
                    "step_id": 1,
                    "description": "Search for information",
                    "tool": "search",
                    "input": {"query": "search terms"},
                    "expected_output": "Search results",
                    "dependencies": []
                },
                {
                    "step_id": 2,
                    "description": "Analyze and summarize findings",
                    "tool": "analyzer",
                    "input": {"data": "search_results"},
                    "expected_output": "Analysis summary",
                    "dependencies": [1]
                }
            ]
        else:
            plan_steps = [
                {
                    "step_id": 1,
                    "description": "Process general task",
                    "tool": None,
                    "input": None,
                    "expected_output": "Task completion",
                    "dependencies": []
                }
            ]
        
        return {
            "task": task,
            "total_steps": len(plan_steps),
            "steps": plan_steps,
            "estimated_duration": len(plan_steps) * 30,  # seconds
            "created_at": datetime.now().isoformat()
        }
    
    def _execute_plan_step(self, step: Dict[str, Any]) -> Any:
        """Execute a single plan step."""
        if step["tool"] and step["tool"] in self.tools:
            # Mock tool execution with dependencies
            if step["dependencies"]:
                # Use results from dependent steps
                dependent_results = []
                for dep_id in step["dependencies"]:
                    if f"step_{dep_id}" in self.execution_state:
                        dependent_results.append(self.execution_state[f"step_{dep_id}"]["result"])
                
                return f"Step {step['step_id']} result using dependencies: {dependent_results}"
            else:
                return f"Step {step['step_id']} result: {step['expected_output']}"
        else:
            return f"Step {step['step_id']} completed without tools"
    
    def _validate_step_result(self, step: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Validate the result of a plan step."""
        # Simple validation logic
        if result and "error" not in str(result).lower():
            return {
                "success": True,
                "status": "Step completed successfully",
                "confidence": 0.9,
                "meets_expectations": True
            }
        else:
            return {
                "success": False,
                "status": "Step failed or produced unexpected result",
                "confidence": 0.1,
                "meets_expectations": False,
                "suggested_action": "Retry with different parameters"
            }
    
    def _adapt_plan(self, failed_step_index: int, validation: Dict[str, Any]):
        """Adapt the plan when a step fails."""
        # Simple adaptation: add a retry step
        failed_step = self.plan["steps"][failed_step_index]
        
        retry_step = {
            "step_id": len(self.plan["steps"]) + 1,
            "description": f"Retry: {failed_step['description']}",
            "tool": failed_step["tool"],
            "input": failed_step["input"],
            "expected_output": failed_step["expected_output"],
            "dependencies": failed_step["dependencies"],
            "retry_of": failed_step["step_id"]
        }
        
        self.plan["steps"].append(retry_step)
        self.plan["total_steps"] += 1
    
    def _synthesize_results(self) -> str:
        """Synthesize final results from all executed steps."""
        successful_steps = [
            step_data for step_data in self.execution_state.values()
            if step_data["validation"]["success"]
        ]
        
        if not successful_steps:
            return "Task could not be completed due to step failures"
        
        return f"Task completed successfully. {len(successful_steps)} steps executed successfully. " \
               f"Results synthesized from: {[s['result'] for s in successful_steps[:3]]}"


# =============================================================================
# 5. Agent Architecture Comparison
# =============================================================================

class AgentArchitectureComparator:
    """
    Compare different agent architectures on various tasks.
    """
    
    def __init__(self):
        self.tools = [
            {"name": "calculator", "description": "Perform calculations"},
            {"name": "search", "description": "Search for information"},
            {"name": "weather", "description": "Get weather information"},
            {"name": "analyzer", "description": "Analyze data"}
        ]
    
    def compare_architectures(self, task: str) -> Dict[str, Any]:
        """
        Compare how different architectures handle the same task.
        
        Args:
            task: Task to execute with all architectures
            
        Returns:
            Comparison results
        """
        # Initialize agents
        react_agent = ReActAgent(self.tools)
        structured_agent = StructuredChatAgent(self.tools)
        plan_execute_agent = PlanAndExecuteAgent(self.tools)
        
        # Execute task with each agent
        results = {}
        
        # ReAct execution
        react_trace = react_agent.execute(task)
        results["react"] = react_trace.get_summary()
        
        # Structured Chat execution
        structured_trace = structured_agent.execute(task)
        results["structured_chat"] = structured_trace.get_summary()
        
        # Plan-and-Execute execution
        plan_execute_trace = plan_execute_agent.execute(task)
        results["plan_and_execute"] = plan_execute_trace.get_summary()
        
        # Generate comparison
        comparison = self._generate_comparison_analysis(results)
        
        return {
            "task": task,
            "architecture_results": results,
            "comparison_analysis": comparison,
            "recommendation": self._recommend_architecture(task, results)
        }
    
    def _generate_comparison_analysis(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate analysis comparing the different architectures."""
        analysis = {
            "execution_complexity": {},
            "tool_usage": {},
            "step_efficiency": {},
            "success_patterns": {}
        }
        
        for arch_name, result in results.items():
            analysis["execution_complexity"][arch_name] = {
                "total_steps": result["total_steps"],
                "tools_used_count": len(result["tools_used"]),
                "complexity_score": result["total_steps"] * len(result["tools_used"])
            }
            
            analysis["tool_usage"][arch_name] = result["tools_used"]
            
            analysis["step_efficiency"][arch_name] = {
                "steps_per_tool": result["total_steps"] / max(1, len(result["tools_used"])),
                "success": result["success"]
            }
        
        return analysis
    
    def _recommend_architecture(self, task: str, results: Dict[str, Dict]) -> Dict[str, str]:
        """Recommend the best architecture for the given task."""
        task_lower = task.lower()
        
        # Simple heuristic-based recommendation
        if "complex" in task_lower or ("calculate" in task_lower and "research" in task_lower):
            recommended = "plan_and_execute"
            reason = "Complex multi-step tasks benefit from explicit planning"
        elif "json" in task_lower or "structured" in task_lower:
            recommended = "structured_chat"
            reason = "Structured data handling requires structured chat agent"
        else:
            recommended = "react"
            reason = "Simple tasks work well with ReAct pattern"
        
        return {
            "recommended_architecture": recommended,
            "reason": reason,
            "alternatives": [arch for arch in results.keys() if arch != recommended]
        }


# =============================================================================
# 6. Demonstration Functions
# =============================================================================

def demonstrate_react_agent():
    """Demonstrate ReAct agent execution."""
    print("🎯 ReAct Agent Demonstration")
    print("=" * 50)
    
    tools = [
        {"name": "calculator", "description": "Perform calculations"},
        {"name": "weather", "description": "Get weather information"}
    ]
    
    agent = ReActAgent(tools)
    
    # Test with different tasks
    tasks = [
        "Calculate 25 + 17",
        "What's the weather like today?",
        "Search for information about Python"
    ]
    
    for task in tasks:
        print(f"\n📝 Task: {task}")
        trace = agent.execute(task)
        summary = trace.get_summary()
        
        print(f"   Steps: {summary['total_steps']}")
        print(f"   Tools used: {summary['tools_used']}")
        print(f"   Result: {summary['final_answer']}")
        print(f"   Success: {summary['success']}")


def demonstrate_structured_chat_agent():
    """Demonstrate Structured Chat agent execution."""
    print("\n🏗️  Structured Chat Agent Demonstration")
    print("=" * 50)
    
    tools = [
        {"name": "calculator", "description": "Advanced calculator"},
        {"name": "search", "description": "Web search tool"},
        {"name": "analyzer", "description": "Data analysis tool"}
    ]
    
    agent = StructuredChatAgent(tools)
    
    task = "Analyze and calculate the average of these numbers: 10, 20, 30, 40, 50"
    context = {
        "user_preferences": {"format": "detailed"},
        "constraints": {"precision": 2}
    }
    
    print(f"\n📝 Task: {task}")
    print(f"📋 Context: {context}")
    
    trace = agent.execute(task, context)
    summary = trace.get_summary()
    
    print(f"\n📊 Results:")
    print(f"   Steps: {summary['total_steps']}")
    print(f"   Tools used: {summary['tools_used']}")
    print(f"   Success: {summary['success']}")


def demonstrate_plan_and_execute_agent():
    """Demonstrate Plan-and-Execute agent execution."""
    print("\n📋 Plan-and-Execute Agent Demonstration")
    print("=" * 50)
    
    tools = [
        {"name": "search", "description": "Information search"},
        {"name": "calculator", "description": "Mathematical operations"},
        {"name": "analyzer", "description": "Data analysis"}
    ]
    
    agent = PlanAndExecuteAgent(tools)
    
    task = "Research Python programming and calculate how many years it has been since its creation"
    
    print(f"\n📝 Task: {task}")
    
    trace = agent.execute(task)
    summary = trace.get_summary()
    
    print(f"\n📊 Results:")
    print(f"   Total steps: {summary['total_steps']}")
    print(f"   Tools used: {summary['tools_used']}")
    print(f"   Duration: {summary['duration_seconds']:.1f}s")
    print(f"   Success: {summary['success']}")


def demonstrate_architecture_comparison():
    """Demonstrate comparison between different architectures."""
    print("\n⚖️  Architecture Comparison Demonstration")
    print("=" * 50)
    
    comparator = AgentArchitectureComparator()
    
    test_task = "Calculate the sum of 15 and 25, then search for information about mathematics"
    
    print(f"\n📝 Test Task: {test_task}")
    
    comparison = comparator.compare_architectures(test_task)
    
    print(f"\n📊 Architecture Performance:")
    for arch_name, result in comparison["architecture_results"].items():
        print(f"   {arch_name.upper()}:")
        print(f"      Steps: {result['total_steps']}")
        print(f"      Tools: {len(result['tools_used'])}")
        print(f"      Success: {result['success']}")
    
    recommendation = comparison["recommendation"]
    print(f"\n💡 Recommendation: {recommendation['recommended_architecture']}")
    print(f"   Reason: {recommendation['reason']}")


# Main demonstration
if __name__ == "__main__":
    print("🎓 Module 4: Agent Architectures")
    print("=" * 60)
    
    # Demonstrate all architectures
    demonstrate_react_agent()
    demonstrate_structured_chat_agent()
    demonstrate_plan_and_execute_agent()
    demonstrate_architecture_comparison()
    
    print("\n✅ Module 4 demonstrations completed!")
    print("Next: Module 5 - Advanced Orchestration with LangGraph")
