"""
Module 5: Advanced Orchestration with LangGraph
==============================================

This module covers advanced agent orchestration using LangGraph for stateful,
multi-agent workflows with memory and complex decision-making patterns.

Learning Objectives:
- Understand LangGraph fundamentals and architecture
- Build stateful agents with memory
- Create multi-agent workflows
- Implement complex orchestration patterns
- Handle conditional flows and loops
"""

import json
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# =============================================================================
# 1. LangGraph State Management
# =============================================================================

class AgentState(TypedDict):
    """
    Represents the state that flows through a LangGraph workflow.
    This is the core data structure that agents operate on.
    """
    messages: List[Dict[str, str]]
    current_task: str
    tools_used: List[str]
    intermediate_results: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    final_answer: Optional[str]
    error_log: List[str]


class NodeType(Enum):
    """Types of nodes in a LangGraph workflow."""
    AGENT = "agent"
    TOOL = "tool" 
    DECISION = "decision"
    MEMORY = "memory"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"


@dataclass
class WorkflowNode:
    """Represents a node in the LangGraph workflow."""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    function: callable
    next_nodes: List[str]
    conditions: Optional[Dict[str, Any]] = None


# =============================================================================
# 2. Simple LangGraph Workflow Simulator
# =============================================================================

class LangGraphSimulator:
    """
    Simulates LangGraph functionality for educational purposes.
    In production, you would use the actual LangGraph library.
    """
    
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.state_history: List[AgentState] = []
        
    def add_node(self, node: WorkflowNode):
        """Add a node to the workflow graph."""
        self.nodes[node.node_id] = node
        self.edges[node.node_id] = node.next_nodes
        
    def add_conditional_edge(self, from_node: str, condition_func: callable, edge_map: Dict[str, str]):
        """Add a conditional edge between nodes."""
        # Store condition logic for the node
        if from_node in self.nodes:
            self.nodes[from_node].conditions = {
                "condition_func": condition_func,
                "edge_map": edge_map
            }
    
    def execute_workflow(self, initial_state: AgentState, start_node: str = "start") -> AgentState:
        """
        Execute the workflow starting from the given node.
        
        Args:
            initial_state: Initial state to begin execution
            start_node: Node ID to start execution from
            
        Returns:
            Final state after workflow completion
        """
        current_state = initial_state.copy()
        current_node_id = start_node
        
        # Add initial state to history
        self.state_history = [current_state.copy()]
        
        while current_node_id and current_state["iteration_count"] < current_state["max_iterations"]:
            if current_node_id not in self.nodes:
                current_state["error_log"].append(f"Node '{current_node_id}' not found")
                break
                
            current_node = self.nodes[current_node_id]
            
            # Execute node function
            try:
                current_state = current_node.function(current_state)
                current_state["iteration_count"] += 1
                
                # Add to history
                self.state_history.append(current_state.copy())
                
                # Determine next node
                next_node = self._get_next_node(current_node, current_state)
                current_node_id = next_node
                
            except Exception as e:
                current_state["error_log"].append(f"Error in node '{current_node_id}': {str(e)}")
                break
        
        return current_state
    
    def _get_next_node(self, node: WorkflowNode, state: AgentState) -> Optional[str]:
        """Determine the next node based on conditions."""
        if node.conditions:
            condition_func = node.conditions["condition_func"]
            edge_map = node.conditions["edge_map"]
            
            # Evaluate condition
            condition_result = condition_func(state)
            return edge_map.get(condition_result)
        
        # Default to first next node if no conditions
        return node.next_nodes[0] if node.next_nodes else None
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get detailed execution trace."""
        trace = []
        for i, state in enumerate(self.state_history):
            trace.append({
                "step": i,
                "iteration": state["iteration_count"],
                "current_task": state["current_task"],
                "tools_used": state["tools_used"],
                "message_count": len(state["messages"]),
                "has_final_answer": state["final_answer"] is not None,
                "error_count": len(state["error_log"])
            })
        return trace


# =============================================================================
# 3. Multi-Agent Workflow Implementation
# =============================================================================

class ResearchAgent:
    """Agent specialized in information gathering and research."""
    
    @staticmethod
    def research_node(state: AgentState) -> AgentState:
        """Research node function."""
        task = state["current_task"]
        
        # Simulate research process
        research_results = {
            "query": task,
            "sources_found": 3,
            "key_findings": [
                f"Finding 1 about {task}",
                f"Finding 2 about {task}", 
                f"Finding 3 about {task}"
            ],
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update state
        state["tools_used"].append("research_tool")
        state["intermediate_results"]["research"] = research_results
        state["messages"].append({
            "role": "research_agent",
            "content": f"Completed research on: {task}"
        })
        
        return state


class AnalysisAgent:
    """Agent specialized in data analysis and processing."""
    
    @staticmethod
    def analysis_node(state: AgentState) -> AgentState:
        """Analysis node function."""
        research_data = state["intermediate_results"].get("research", {})
        
        # Simulate analysis process
        analysis_results = {
            "data_processed": True,
            "insights_generated": 2,
            "key_insights": [
                "Primary insight from analysis",
                "Secondary insight from analysis"
            ],
            "statistical_summary": {
                "confidence_score": research_data.get("confidence", 0.5),
                "data_quality": "high"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Update state
        state["tools_used"].append("analysis_tool")
        state["intermediate_results"]["analysis"] = analysis_results
        state["messages"].append({
            "role": "analysis_agent", 
            "content": "Completed data analysis and insight generation"
        })
        
        return state


class SynthesisAgent:
    """Agent specialized in synthesizing results from multiple sources."""
    
    @staticmethod
    def synthesis_node(state: AgentState) -> AgentState:
        """Synthesis node function."""
        research = state["intermediate_results"].get("research", {})
        analysis = state["intermediate_results"].get("analysis", {})
        
        # Synthesize final answer
        final_answer = f"""
        Task: {state['current_task']}
        
        Research Summary:
        - Sources analyzed: {research.get('sources_found', 0)}
        - Key findings: {len(research.get('key_findings', []))}
        
        Analysis Summary:
        - Insights generated: {analysis.get('insights_generated', 0)}
        - Confidence: {analysis.get('statistical_summary', {}).get('confidence_score', 0)}
        
        Conclusion: Based on the research and analysis, I have synthesized a comprehensive response to the task.
        """
        
        # Update state
        state["tools_used"].append("synthesis_tool")
        state["final_answer"] = final_answer.strip()
        state["messages"].append({
            "role": "synthesis_agent",
            "content": "Completed final synthesis"
        })
        
        return state


# =============================================================================
# 4. Decision Making and Routing
# =============================================================================

class WorkflowRouter:
    """Handles conditional routing in LangGraph workflows."""
    
    @staticmethod
    def task_classifier(state: AgentState) -> str:
        """
        Classify the task to determine workflow path.
        
        Returns:
            Route name for conditional edge
        """
        task = state["current_task"].lower()
        
        if "research" in task or "search" in task or "find" in task:
            return "research_path"
        elif "calculate" in task or "analyze" in task or "compute" in task:
            return "analysis_path"
        elif "complex" in task or ("research" in task and "analyze" in task):
            return "multi_agent_path"
        else:
            return "simple_path"
    
    @staticmethod
    def quality_checker(state: AgentState) -> str:
        """
        Check if results meet quality standards.
        
        Returns:
            'continue' or 'retry' or 'complete'
        """
        error_count = len(state["error_log"])
        tools_used = len(state["tools_used"])
        has_final_answer = state["final_answer"] is not None
        
        if error_count > 2:
            return "retry"
        elif has_final_answer and tools_used >= 2:
            return "complete"
        elif state["iteration_count"] >= state["max_iterations"]:
            return "complete"
        else:
            return "continue"
    
    @staticmethod
    def memory_manager(state: AgentState) -> AgentState:
        """Manage memory and context across iterations."""
        # Summarize if too many messages
        if len(state["messages"]) > 10:
            summary = {
                "role": "system",
                "content": f"Summary: {len(state['messages'])} messages processed, " + 
                          f"{len(state['tools_used'])} tools used"
            }
            # Keep first 2, last 5, and summary
            state["messages"] = (state["messages"][:2] + 
                               [summary] + 
                               state["messages"][-5:])
        
        return state


# =============================================================================
# 5. Complete Workflow Examples
# =============================================================================

class ComplexWorkflowBuilder:
    """Builds complex multi-agent workflows using LangGraph patterns."""
    
    def __init__(self):
        self.simulator = LangGraphSimulator()
        
    def build_research_analysis_workflow(self) -> LangGraphSimulator:
        """Build a workflow that combines research and analysis."""
        
        # Create workflow nodes
        nodes = [
            WorkflowNode(
                node_id="start",
                node_type=NodeType.DECISION,
                name="Task Router",
                description="Route task to appropriate workflow",
                function=self._start_node,
                next_nodes=[]  # Will be set by conditional edges
            ),
            WorkflowNode(
                node_id="research",
                node_type=NodeType.AGENT,
                name="Research Agent",
                description="Gather information and data",
                function=ResearchAgent.research_node,
                next_nodes=["analysis"]
            ),
            WorkflowNode(
                node_id="analysis", 
                node_type=NodeType.AGENT,
                name="Analysis Agent",
                description="Analyze gathered data",
                function=AnalysisAgent.analysis_node,
                next_nodes=["quality_check"]
            ),
            WorkflowNode(
                node_id="synthesis",
                node_type=NodeType.AGENT,
                name="Synthesis Agent", 
                description="Synthesize final results",
                function=SynthesisAgent.synthesis_node,
                next_nodes=["memory_update"]
            ),
            WorkflowNode(
                node_id="quality_check",
                node_type=NodeType.VALIDATOR,
                name="Quality Checker",
                description="Validate result quality",
                function=self._quality_check_node,
                next_nodes=[]  # Conditional
            ),
            WorkflowNode(
                node_id="memory_update",
                node_type=NodeType.MEMORY,
                name="Memory Manager",
                description="Update workflow memory",
                function=WorkflowRouter.memory_manager,
                next_nodes=["end"]
            ),
            WorkflowNode(
                node_id="end",
                node_type=NodeType.DECISION,
                name="End Node",
                description="Workflow completion",
                function=self._end_node,
                next_nodes=[]
            )
        ]
        
        # Add nodes to simulator
        for node in nodes:
            self.simulator.add_node(node)
        
        # Add conditional edges
        self.simulator.add_conditional_edge(
            "start",
            WorkflowRouter.task_classifier,
            {
                "research_path": "research",
                "analysis_path": "analysis", 
                "multi_agent_path": "research",
                "simple_path": "synthesis"
            }
        )
        
        self.simulator.add_conditional_edge(
            "quality_check",
            WorkflowRouter.quality_checker,
            {
                "continue": "synthesis",
                "retry": "research",
                "complete": "synthesis"
            }
        )
        
        return self.simulator
    
    def _start_node(self, state: AgentState) -> AgentState:
        """Initialize workflow state."""
        state["messages"].append({
            "role": "system",
            "content": f"Starting workflow for task: {state['current_task']}"
        })
        return state
    
    def _quality_check_node(self, state: AgentState) -> AgentState:
        """Perform quality validation."""
        quality_score = len(state["tools_used"]) * 0.3 + len(state["intermediate_results"]) * 0.4
        
        state["intermediate_results"]["quality_check"] = {
            "score": quality_score,
            "passed": quality_score >= 1.0,
            "timestamp": datetime.now().isoformat()
        }
        
        return state
    
    def _end_node(self, state: AgentState) -> AgentState:
        """Finalize workflow execution."""
        state["messages"].append({
            "role": "system",
            "content": "Workflow completed successfully"
        })
        return state


# =============================================================================
# 6. Demonstration Functions
# =============================================================================

def demonstrate_simple_workflow():
    """Demonstrate a simple LangGraph workflow."""
    print("🔄 Simple LangGraph Workflow Demonstration")
    print("=" * 50)
    
    # Create initial state
    initial_state: AgentState = {
        "messages": [],
        "current_task": "Research machine learning applications in healthcare",
        "tools_used": [],
        "intermediate_results": {},
        "iteration_count": 0,
        "max_iterations": 5,
        "final_answer": None,
        "error_log": []
    }
    
    # Build and execute workflow
    builder = ComplexWorkflowBuilder()
    workflow = builder.build_research_analysis_workflow()
    
    print(f"📋 Task: {initial_state['current_task']}")
    
    # Execute workflow
    final_state = workflow.execute_workflow(initial_state, "start")
    
    # Display results
    print(f"\n📊 Workflow Results:")
    print(f"   Iterations: {final_state['iteration_count']}")
    print(f"   Tools used: {len(final_state['tools_used'])}")
    print(f"   Messages: {len(final_state['messages'])}")
    print(f"   Errors: {len(final_state['error_log'])}")
    print(f"   Completed: {'Yes' if final_state['final_answer'] else 'No'}")
    
    if final_state['final_answer']:
        print(f"\n✅ Final Answer:")
        print(final_state['final_answer'][:200] + "...")


def demonstrate_execution_trace():
    """Demonstrate workflow execution tracing."""
    print("\n📊 Execution Trace Demonstration")
    print("=" * 50)
    
    # Create workflow for tracing
    builder = ComplexWorkflowBuilder()
    workflow = builder.build_research_analysis_workflow()
    
    initial_state: AgentState = {
        "messages": [],
        "current_task": "Analyze stock market trends and provide investment insights",
        "tools_used": [],
        "intermediate_results": {},
        "iteration_count": 0,
        "max_iterations": 4,
        "final_answer": None,
        "error_log": []
    }
    
    # Execute and trace
    workflow.execute_workflow(initial_state, "start")
    trace = workflow.get_execution_trace()
    
    print("🔍 Execution Trace:")
    for step in trace:
        print(f"   Step {step['step']}: Iteration {step['iteration']}")
        print(f"      Tools: {step['tools_used']}")
        print(f"      Messages: {step['message_count']}")
        print(f"      Status: {'Complete' if step['has_final_answer'] else 'In Progress'}")


def demonstrate_conditional_routing():
    """Demonstrate conditional routing in workflows."""
    print("\n🔀 Conditional Routing Demonstration") 
    print("=" * 50)
    
    test_tasks = [
        "Research the latest developments in quantum computing",
        "Calculate the ROI for a new marketing campaign",
        "Research and analyze the impact of AI on employment",
        "What is the weather today?"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{i}. Task: {task}")
        
        # Create state for routing test
        test_state: AgentState = {
            "messages": [],
            "current_task": task,
            "tools_used": [],
            "intermediate_results": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "final_answer": None,
            "error_log": []
        }
        
        # Test routing
        route = WorkflowRouter.task_classifier(test_state)
        print(f"   Routed to: {route}")


def demonstrate_multi_agent_coordination():
    """Demonstrate multi-agent coordination."""
    print("\n🤝 Multi-Agent Coordination Demonstration")
    print("=" * 50)
    
    # Simulate multi-agent state
    state: AgentState = {
        "messages": [],
        "current_task": "Comprehensive market analysis for tech stocks",
        "tools_used": [],
        "intermediate_results": {},
        "iteration_count": 0,
        "max_iterations": 10,
        "final_answer": None,
        "error_log": []
    }
    
    print(f"📋 Multi-Agent Task: {state['current_task']}")
    
    # Research Agent
    print(f"\n🔍 Research Agent Working...")
    state = ResearchAgent.research_node(state)
    print(f"   Research completed: {state['intermediate_results']['research']['sources_found']} sources")
    
    # Analysis Agent  
    print(f"\n📊 Analysis Agent Working...")
    state = AnalysisAgent.analysis_node(state)
    print(f"   Analysis completed: {state['intermediate_results']['analysis']['insights_generated']} insights")
    
    # Synthesis Agent
    print(f"\n🔗 Synthesis Agent Working...")
    state = SynthesisAgent.synthesis_node(state)
    print(f"   Synthesis completed: Final answer generated")
    
    print(f"\n✅ Multi-agent coordination complete!")
    print(f"   Total agents involved: 3")
    print(f"   Tools used: {len(state['tools_used'])}")
    print(f"   Messages exchanged: {len(state['messages'])}")


# Main demonstration
if __name__ == "__main__":
    print("🎓 Module 5: Advanced Orchestration with LangGraph")
    print("=" * 60)
    
    # Demonstrate all LangGraph concepts
    demonstrate_simple_workflow()
    demonstrate_execution_trace()
    demonstrate_conditional_routing()
    demonstrate_multi_agent_coordination()
    
    print("\n✅ Module 5 demonstrations completed!")
    print("Next: Module 6 - Real-World Applications")
