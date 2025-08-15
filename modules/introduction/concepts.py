"""
Module 1: Introduction to AI Agents
==================================

This module covers the fundamental concepts of AI agents and their differences
from standalone LLMs.

Learning Objectives:
- Understand what AI agents are and how they work
- Learn the key differences between agents and LLMs
- Identify when to use agents vs other approaches
- Explore basic agent components (LLM + Tools + Memory)
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class AgentCapability(Enum):
    """Enumeration of core agent capabilities."""
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    PLANNING = "planning"


@dataclass
class AgentCharacteristics:
    """Data class defining key characteristics of AI agents."""
    has_tools: bool
    maintains_memory: bool
    can_plan: bool
    accesses_real_time_data: bool
    is_stateful: bool
    
    def compare_to_llm(self) -> Dict[str, str]:
        """
        Compare agent characteristics to standalone LLM.
        
        Returns:
            Dict mapping characteristics to comparisons
        """
        return {
            "Action Capability": "Takes actions using tools" if self.has_tools else "Generates text only",
            "Data Access": "Real-time information" if self.accesses_real_time_data else "Limited to training data",
            "State Management": "Maintains context" if self.is_stateful else "Stateless",
            "Memory": "Persistent memory" if self.maintains_memory else "No memory between calls",
            "Planning": "Can plan multi-step tasks" if self.can_plan else "Single response generation"
        }


def when_to_use_agents() -> Dict[str, List[str]]:
    """
    Guidelines for when to use agents vs alternatives.
    
    Returns:
        Dictionary with use cases and avoid cases
    """
    return {
        "best_for": [
            "Dynamic tasks requiring real-world interaction",
            "Multi-step research and analysis",
            "Tasks needing tool orchestration",
            "Interactive problem-solving sessions",
            "Scenarios requiring context preservation",
            "Complex decision-making workflows"
        ],
        "avoid_for": [
            "Simple, deterministic workflows",
            "Single-shot text generation",
            "Highly predictable processes",
            "Tasks with strict latency requirements",
            "Simple classification/extraction tasks",
            "Scenarios where explainability is critical"
        ]
    }


def demonstrate_agent_components():
    """
    Demonstrate the three core components of AI agents.
    
    This is a conceptual demonstration showing how LLM + Tools + Memory
    work together to create an intelligent agent.
    """
    print("🤖 AI Agent Components Demonstration")
    print("=" * 50)
    
    # Component 1: LLM (Language Model)
    print("\n1. 🧠 LLM Component:")
    print("   - Provides reasoning and language understanding")
    print("   - Generates responses and plans")
    print("   - Decides which tools to use")
    
    # Component 2: Tools
    print("\n2. 🔧 Tools Component:")
    print("   - External APIs (weather, stock prices, search)")
    print("   - Databases and file systems")
    print("   - Calculators and specialized functions")
    print("   - Code execution environments")
    
    # Component 3: Memory
    print("\n3. 💾 Memory Component:")
    print("   - Conversation history")
    print("   - Previous tool results")
    print("   - Learned patterns and preferences")
    print("   - Context across multiple interactions")
    
    print("\n🔄 How they work together:")
    print("   User Input → LLM analyzes → Selects tools → Executes → Updates memory → Responds")


class SimpleAgentSimulator:
    """
    A simple simulator to demonstrate agent thinking process.
    
    This class simulates how an agent would approach a multi-step task,
    showing the reasoning, tool selection, and memory usage.
    """
    
    def __init__(self):
        self.memory: List[str] = []
        self.available_tools = [
            "web_search", "calculator", "weather_api", 
            "email_sender", "calendar", "note_taker"
        ]
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Simulate agent processing of a user request.
        
        Args:
            user_input: The user's request
            
        Returns:
            Dictionary showing agent's thinking process
        """
        # Add to memory
        self.memory.append(f"User: {user_input}")
        
        # Simulate reasoning process
        thinking_steps = self._simulate_reasoning(user_input)
        
        # Simulate tool selection
        selected_tools = self._simulate_tool_selection(user_input)
        
        # Simulate response generation
        response = self._simulate_response(user_input, selected_tools)
        
        # Update memory
        self.memory.append(f"Agent: {response}")
        
        return {
            "user_input": user_input,
            "thinking_steps": thinking_steps,
            "tools_selected": selected_tools,
            "response": response,
            "memory_items": len(self.memory)
        }
    
    def _simulate_reasoning(self, input_text: str) -> List[str]:
        """Simulate the agent's reasoning process."""
        if "weather" in input_text.lower():
            return [
                "User is asking about weather",
                "I need current weather data",
                "Should use weather_api tool",
                "May need location information"
            ]
        elif any(word in input_text.lower() for word in ["calculate", "math", "compute"]):
            return [
                "User needs mathematical computation",
                "Should use calculator tool",
                "Parse numbers and operation",
                "Return calculated result"
            ]
        else:
            return [
                "Analyzing user request",
                "Determining appropriate response strategy",
                "Selecting relevant tools",
                "Planning response structure"
            ]
    
    def _simulate_tool_selection(self, input_text: str) -> List[str]:
        """Simulate tool selection based on input."""
        tools = []
        if "weather" in input_text.lower():
            tools.append("weather_api")
        if any(word in input_text.lower() for word in ["calculate", "math"]):
            tools.append("calculator")
        if "search" in input_text.lower():
            tools.append("web_search")
        
        return tools if tools else ["general_knowledge"]
    
    def _simulate_response(self, input_text: str, tools: List[str]) -> str:
        """Simulate response generation."""
        if "weather_api" in tools:
            return f"I'll check the weather for you using the weather API. [Tool: weather_api executed]"
        elif "calculator" in tools:
            return f"Let me calculate that for you. [Tool: calculator executed]"
        else:
            return f"I understand your request. Let me help you with that using my available tools: {', '.join(tools)}"


# Example usage and demonstrations
if __name__ == "__main__":
    print("🎓 Module 1: Introduction to AI Agents")
    print("=" * 60)
    
    # Demonstrate agent characteristics
    print("\n📊 Agent vs LLM Comparison:")
    agent_chars = AgentCharacteristics(
        has_tools=True,
        maintains_memory=True,
        can_plan=True,
        accesses_real_time_data=True,
        is_stateful=True
    )
    
    comparison = agent_chars.compare_to_llm()
    for aspect, description in comparison.items():
        print(f"   {aspect}: {description}")
    
    # Show when to use agents
    print("\n📋 When to Use Agents:")
    guidelines = when_to_use_agents()
    
    print("\n   ✅ Best for:")
    for use_case in guidelines["best_for"]:
        print(f"      • {use_case}")
    
    print("\n   ❌ Avoid for:")
    for avoid_case in guidelines["avoid_for"]:
        print(f"      • {avoid_case}")
    
    # Demonstrate components
    print("\n" + "=" * 60)
    demonstrate_agent_components()
    
    # Simulate agent interaction
    print("\n" + "=" * 60)
    print("🎮 Agent Simulation Demo:")
    
    simulator = SimpleAgentSimulator()
    
    test_requests = [
        "What's the weather like in New York?",
        "Calculate 25 * 47 + 129",
        "Search for information about machine learning"
    ]
    
    for request in test_requests:
        print(f"\n📝 Processing: '{request}'")
        result = simulator.process_request(request)
        
        print(f"   🤔 Thinking: {result['thinking_steps'][0]}")
        print(f"   🔧 Tools: {', '.join(result['tools_selected'])}")
        print(f"   💬 Response: {result['response']}")
        print(f"   💾 Memory items: {result['memory_items']}")
