"""
Module 2: Tool Calling & Function Calling
=========================================

This module covers how AI agents interact with external tools and functions.
Learn about tool schemas, execution patterns, and best practices.

Learning Objectives:
- Understand tool calling mechanisms
- Learn to define tool schemas properly
- Compare traditional vs embedded tool calling
- Implement practical tool examples
- Handle tool execution and error cases
"""

import json
import requests
import random
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# =============================================================================
# 1. Basic Tool Definition Examples
# =============================================================================

@tool
def simple_calculator(expression: str) -> str:
    """
    Evaluate a simple mathematical expression.
    
    Args:
        expression: Mathematical expression as string (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the calculation as string
        
    Example:
        simple_calculator("10 + 5") → "15"
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather_api_mock(location: str) -> Dict[str, Any]:
    """
    Fetch current weather for a location (mock implementation).
    
    Args:
        location: City name or location (e.g., "New York", "London")
        
    Returns:
        Dictionary with weather information
        
    Example:
        weather_api_mock("New York") → {"temp": 72, "condition": "sunny"}
    """
    # Mock weather data for demonstration
    mock_weather = {
        "new york": {"temperature": 72, "condition": "sunny", "humidity": 45},
        "london": {"temperature": 58, "condition": "cloudy", "humidity": 78},
        "tokyo": {"temperature": 68, "condition": "rainy", "humidity": 85},
        "paris": {"temperature": 65, "condition": "partly cloudy", "humidity": 60}
    }
    
    location_key = location.lower()
    if location_key in mock_weather:
        return {
            "location": location,
            "data": mock_weather[location_key],
            "source": "mock_api"
        }
    else:
        # Generate random weather for unknown locations
        return {
            "location": location,
            "data": {
                "temperature": random.randint(50, 85),
                "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
                "humidity": random.randint(30, 90)
            },
            "source": "mock_api"
        }


@tool
def text_analyzer(text: str) -> Dict[str, Any]:
    """
    Analyze text and return statistics.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with text analysis results
        
    Example:
        text_analyzer("Hello world") → {"words": 2, "chars": 11, "sentences": 1}
    """
    words = len(text.split())
    characters = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "word_count": words,
        "character_count": characters,
        "sentence_count": max(1, sentences),  # At least 1 sentence
        "avg_word_length": round(characters / max(1, words), 2)
    }


# =============================================================================
# 2. Advanced Tool Classes
# =============================================================================

class NumberListInput(BaseModel):
    """Input schema for tools that work with number lists."""
    numbers: List[float] = Field(description="List of numbers to process")


class MathOperationsTool(BaseTool):
    """Advanced tool for mathematical operations on lists of numbers."""
    
    name: str = "math_operations"
    description: str = "Perform mathematical operations on lists of numbers"
    
    def _run(self, operation: str, numbers: List[float]) -> Dict[str, Any]:
        """
        Execute mathematical operation on number list.
        
        Args:
            operation: Type of operation (sum, product, average, min, max)
            numbers: List of numbers to process
            
        Returns:
            Dictionary with operation result and metadata
        """
        if not numbers:
            return {"error": "Empty number list provided"}
        
        try:
            if operation == "sum":
                result = sum(numbers)
            elif operation == "product":
                result = 1
                for num in numbers:
                    result *= num
            elif operation == "average":
                result = sum(numbers) / len(numbers)
            elif operation == "min":
                result = min(numbers)
            elif operation == "max":
                result = max(numbers)
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            return {
                "operation": operation,
                "input_numbers": numbers,
                "result": result,
                "count": len(numbers)
            }
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}


# =============================================================================
# 3. Tool Calling Flow Demonstration
# =============================================================================

class ToolCallFlow:
    """
    Demonstrates the complete tool calling flow from user input to response.
    
    This class simulates how an LLM would process tool calls in a real agent system.
    """
    
    def __init__(self):
        self.tools = {
            "calculator": simple_calculator,
            "weather": weather_api_mock,
            "text_analyzer": text_analyzer
        }
        self.call_history: List[Dict] = []
    
    def simulate_tool_call(self, user_input: str) -> Dict[str, Any]:
        """
        Simulate the complete tool calling process.
        
        Args:
            user_input: User's natural language request
            
        Returns:
            Dictionary containing the complete flow simulation
        """
        # Step 1: Parse user intent
        intent = self._parse_user_intent(user_input)
        
        # Step 2: Select appropriate tool
        tool_selection = self._select_tool(intent)
        
        # Step 3: Extract parameters
        parameters = self._extract_parameters(user_input, tool_selection)
        
        # Step 4: Execute tool
        tool_result = self._execute_tool(tool_selection, parameters)
        
        # Step 5: Generate response
        response = self._generate_response(tool_result, user_input)
        
        # Log the interaction
        interaction = {
            "user_input": user_input,
            "intent": intent,
            "tool_selected": tool_selection,
            "parameters": parameters,
            "tool_result": tool_result,
            "final_response": response,
            "timestamp": len(self.call_history)
        }
        self.call_history.append(interaction)
        
        return interaction
    
    def _parse_user_intent(self, user_input: str) -> str:
        """Parse user input to determine intent."""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["calculate", "compute", "math", "+", "-", "*", "/"]):
            return "calculation"
        elif any(word in input_lower for word in ["weather", "temperature", "forecast"]):
            return "weather_query"
        elif any(word in input_lower for word in ["analyze", "count", "words", "text"]):
            return "text_analysis"
        else:
            return "general"
    
    def _select_tool(self, intent: str) -> str:
        """Select appropriate tool based on intent."""
        tool_mapping = {
            "calculation": "calculator",
            "weather_query": "weather",
            "text_analysis": "text_analyzer"
        }
        return tool_mapping.get(intent, "calculator")
    
    def _extract_parameters(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """Extract parameters for the selected tool."""
        if tool_name == "calculator":
            # Extract mathematical expression
            import re
            math_pattern = r'[\d+\-*/().\s]+'
            matches = re.findall(math_pattern, user_input)
            expression = matches[0].strip() if matches else "1+1"
            return {"expression": expression}
        
        elif tool_name == "weather":
            # Extract location
            words = user_input.split()
            # Simple heuristic: look for city names (capitalized words)
            locations = [word for word in words if word.istitle() and len(word) > 2]
            location = locations[0] if locations else "New York"
            return {"location": location}
        
        elif tool_name == "text_analyzer":
            # Use the input text itself
            return {"text": user_input}
        
        return {}
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute the selected tool with parameters."""
        if tool_name in self.tools:
            try:
                tool_func = self.tools[tool_name]
                return tool_func.invoke(parameters) if hasattr(tool_func, 'invoke') else tool_func(**parameters)
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}"}
        else:
            return {"error": f"Tool {tool_name} not found"}
    
    def _generate_response(self, tool_result: Any, original_input: str) -> str:
        """Generate human-readable response from tool result."""
        if isinstance(tool_result, dict) and "error" in tool_result:
            return f"I encountered an error: {tool_result['error']}"
        
        if isinstance(tool_result, dict):
            if "temperature" in str(tool_result):
                # Weather response
                data = tool_result.get("data", {})
                location = tool_result.get("location", "Unknown")
                return f"The weather in {location} is {data.get('temperature', 'unknown')}°F and {data.get('condition', 'unknown')}."
            
            elif "word_count" in tool_result:
                # Text analysis response
                return f"Your text has {tool_result['word_count']} words, {tool_result['character_count']} characters, and {tool_result['sentence_count']} sentences."
        
        # Default response for calculations or other results
        return f"The result is: {tool_result}"


# =============================================================================
# 4. Traditional vs Embedded Tool Calling Comparison
# =============================================================================

class TraditionalToolCalling:
    """
    Simulates traditional tool calling where the client handles execution.
    Higher risk of hallucination but more control.
    """
    
    def __init__(self):
        self.tool_registry = {
            "get_weather": {
                "description": "Get weather for a location",
                "parameters": {"location": "string"}
            },
            "calculate": {
                "description": "Perform calculation",
                "parameters": {"expression": "string"}
            }
        }
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process request using traditional approach."""
        # LLM generates tool call suggestion
        tool_call_suggestion = self._llm_generate_tool_call(user_input)
        
        # Client must validate and execute
        if self._validate_tool_call(tool_call_suggestion):
            result = self._execute_tool_call(tool_call_suggestion)
            return {
                "approach": "traditional",
                "tool_call": tool_call_suggestion,
                "result": result,
                "risk_level": "higher_hallucination_risk"
            }
        else:
            return {
                "approach": "traditional",
                "error": "Invalid tool call generated",
                "risk_level": "higher_hallucination_risk"
            }
    
    def _llm_generate_tool_call(self, user_input: str) -> Dict[str, Any]:
        """Simulate LLM generating a tool call."""
        if "weather" in user_input.lower():
            return {
                "tool": "get_weather",
                "parameters": {"location": "New York"}  # Simplified extraction
            }
        elif "calculate" in user_input.lower():
            return {
                "tool": "calculate",
                "parameters": {"expression": "2+2"}  # Simplified extraction
            }
        else:
            return {"tool": "unknown", "parameters": {}}
    
    def _validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """Validate the generated tool call."""
        tool_name = tool_call.get("tool")
        return tool_name in self.tool_registry
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute the validated tool call."""
        tool_name = tool_call["tool"]
        params = tool_call["parameters"]
        
        if tool_name == "get_weather":
            return weather_api_mock.invoke(params)
        elif tool_name == "calculate":
            return simple_calculator.invoke(params)
        else:
            return "Tool not implemented"


class EmbeddedToolCalling:
    """
    Simulates embedded tool calling where the library manages execution.
    Built-in validation and retries, lower hallucination risk.
    """
    
    def __init__(self):
        self.tools = [simple_calculator, weather_api_mock, text_analyzer]
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process request using embedded approach."""
        # Library automatically handles tool selection and execution
        flow = ToolCallFlow()
        result = flow.simulate_tool_call(user_input)
        
        return {
            "approach": "embedded",
            "automatic_tool_selection": True,
            "built_in_validation": True,
            "result": result,
            "risk_level": "lower_hallucination_risk"
        }


# =============================================================================
# 5. Demonstration and Testing Functions
# =============================================================================

def demonstrate_basic_tools():
    """Demonstrate basic tool usage."""
    print("🔧 Basic Tool Demonstrations")
    print("=" * 50)
    
    # Calculator tool
    print("\n📊 Calculator Tool:")
    calc_result = simple_calculator.invoke({"expression": "15 + 27 * 2"})
    print(f"   Expression: 15 + 27 * 2")
    print(f"   Result: {calc_result}")
    
    # Weather tool
    print("\n🌤️  Weather Tool:")
    weather_result = weather_api_mock.invoke({"location": "Tokyo"})
    print(f"   Location: Tokyo")
    print(f"   Result: {weather_result}")
    
    # Text analyzer tool
    print("\n📝 Text Analyzer Tool:")
    text_result = text_analyzer.invoke({"text": "This is a sample text for analysis."})
    print(f"   Text: 'This is a sample text for analysis.'")
    print(f"   Result: {text_result}")


def demonstrate_tool_calling_flow():
    """Demonstrate the complete tool calling flow."""
    print("\n🔄 Tool Calling Flow Demonstration")
    print("=" * 50)
    
    flow = ToolCallFlow()
    
    test_inputs = [
        "What's the weather like in Paris?",
        "Calculate 125 + 75 * 3",
        "Analyze this text: LangChain makes it easy to build AI agents!"
    ]
    
    for user_input in test_inputs:
        print(f"\n📝 Input: '{user_input}'")
        result = flow.simulate_tool_call(user_input)
        
        print(f"   🎯 Intent: {result['intent']}")
        print(f"   🔧 Tool: {result['tool_selected']}")
        print(f"   📊 Parameters: {result['parameters']}")
        print(f"   💬 Response: {result['final_response']}")


def compare_calling_approaches():
    """Compare traditional vs embedded tool calling."""
    print("\n⚖️  Traditional vs Embedded Tool Calling")
    print("=" * 50)
    
    traditional = TraditionalToolCalling()
    embedded = EmbeddedToolCalling()
    
    test_input = "What's the weather in London?"
    
    print(f"\n📝 Test Input: '{test_input}'")
    
    # Traditional approach
    print("\n🔧 Traditional Approach:")
    trad_result = traditional.process_request(test_input)
    print(f"   Approach: {trad_result['approach']}")
    print(f"   Risk Level: {trad_result['risk_level']}")
    print(f"   Tool Call: {trad_result.get('tool_call', 'N/A')}")
    
    # Embedded approach
    print("\n🔧 Embedded Approach:")
    emb_result = embedded.process_request(test_input)
    print(f"   Approach: {emb_result['approach']}")
    print(f"   Risk Level: {emb_result['risk_level']}")
    print(f"   Auto Selection: {emb_result['automatic_tool_selection']}")
    print(f"   Built-in Validation: {emb_result['built_in_validation']}")


# Example usage
if __name__ == "__main__":
    print("🎓 Module 2: Tool Calling & Function Calling")
    print("=" * 60)
    
    # Basic tool demonstrations
    demonstrate_basic_tools()
    
    # Tool calling flow
    demonstrate_tool_calling_flow()
    
    # Compare approaches
    compare_calling_approaches()
    
    print("\n✅ Module 2 demonstrations completed!")
    print("Next: Module 3 - Building Custom Tools")
