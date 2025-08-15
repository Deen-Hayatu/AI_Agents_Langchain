"""
Module 3: Building Custom Tools
==============================

This module focuses on creating sophisticated custom tools for AI agents.
Learn to build math toolkits, hybrid tools, and specialized tool collections.

Learning Objectives:
- Build comprehensive math toolkits
- Create hybrid tools combining multiple capabilities
- Implement error handling and validation
- Design reusable tool architectures
- Integrate with external APIs and services
"""

import numpy as np
import requests
import json
import math
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta


# =============================================================================
# 1. Advanced Math Toolkit
# =============================================================================

@tool
def add_numbers(numbers: List[float]) -> Dict[str, Any]:
    """
    Add a list of numbers with detailed result information.
    
    Args:
        numbers: List of numbers to add
        
    Returns:
        Dictionary with sum and metadata
        
    Example:
        add_numbers([1, 2, 3, 4]) → {"sum": 10, "count": 4, "average": 2.5}
    """
    if not numbers:
        return {"error": "Empty list provided", "sum": 0}
    
    try:
        total = sum(numbers)
        return {
            "operation": "addition",
            "input_numbers": numbers,
            "sum": total,
            "count": len(numbers),
            "average": total / len(numbers),
            "min_value": min(numbers),
            "max_value": max(numbers)
        }
    except Exception as e:
        return {"error": f"Addition failed: {str(e)}"}


@tool
def multiply_numbers(numbers: List[float]) -> Dict[str, Any]:
    """
    Multiply a list of numbers with comprehensive analysis.
    
    Args:
        numbers: List of numbers to multiply
        
    Returns:
        Dictionary with product and analysis
        
    Example:
        multiply_numbers([2, 3, 4]) → {"product": 24, "count": 3, "geometric_mean": 2.88}
    """
    if not numbers:
        return {"error": "Empty list provided", "product": 1}
    
    try:
        product = float(np.prod(numbers))
        geometric_mean = math.pow(abs(product), 1/len(numbers)) if product != 0 else 0
        
        return {
            "operation": "multiplication",
            "input_numbers": numbers,
            "product": product,
            "count": len(numbers),
            "geometric_mean": round(geometric_mean, 4),
            "contains_zero": 0 in numbers,
            "all_positive": all(n > 0 for n in numbers)
        }
    except Exception as e:
        return {"error": f"Multiplication failed: {str(e)}"}


@tool
def statistical_analysis(numbers: List[float]) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with statistical measures
        
    Example:
        statistical_analysis([1,2,3,4,5]) → {"mean": 3, "median": 3, "std": 1.58}
    """
    if not numbers:
        return {"error": "Empty list provided"}
    
    try:
        np_array = np.array(numbers)
        
        return {
            "operation": "statistical_analysis",
            "input_numbers": numbers,
            "count": len(numbers),
            "mean": float(np.mean(np_array)),
            "median": float(np.median(np_array)),
            "std_deviation": float(np.std(np_array)),
            "variance": float(np.var(np_array)),
            "min": float(np.min(np_array)),
            "max": float(np.max(np_array)),
            "range": float(np.max(np_array) - np.min(np_array)),
            "percentile_25": float(np.percentile(np_array, 25)),
            "percentile_75": float(np.percentile(np_array, 75))
        }
    except Exception as e:
        return {"error": f"Statistical analysis failed: {str(e)}"}


@tool
def matrix_operations(matrix_a: List[List[float]], matrix_b: List[List[float]], operation: str) -> Dict[str, Any]:
    """
    Perform matrix operations (add, multiply, subtract).
    
    Args:
        matrix_a: First matrix as list of lists
        matrix_b: Second matrix as list of lists  
        operation: Operation to perform ('add', 'multiply', 'subtract')
        
    Returns:
        Dictionary with operation result and metadata
        
    Example:
        matrix_operations([[1,2],[3,4]], [[5,6],[7,8]], "add") → result matrix and info
    """
    try:
        np_a = np.array(matrix_a)
        np_b = np.array(matrix_b)
        
        if operation == "add":
            result = np_a + np_b
        elif operation == "subtract":
            result = np_a - np_b
        elif operation == "multiply":
            result = np.dot(np_a, np_b)
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "operation": f"matrix_{operation}",
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "result": result.tolist(),
            "result_shape": result.shape,
            "input_shapes": {"a": np_a.shape, "b": np_b.shape}
        }
    except Exception as e:
        return {"error": f"Matrix operation failed: {str(e)}"}


# =============================================================================
# 2. Research and Information Tools
# =============================================================================

@tool
def web_search_mock(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Mock web search tool that simulates search results.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
        
    Returns:
        Dictionary with mock search results
        
    Example:
        web_search_mock("Python programming") → {"results": [...], "query": "Python programming"}
    """
    # Mock search results for common queries
    mock_results = {
        "python": [
            {"title": "Python.org", "url": "https://python.org", "snippet": "Official Python website"},
            {"title": "Python Tutorial", "url": "https://docs.python.org/tutorial", "snippet": "Learn Python programming"},
            {"title": "Real Python", "url": "https://realpython.com", "snippet": "Python tutorials and articles"}
        ],
        "machine learning": [
            {"title": "Scikit-learn", "url": "https://scikit-learn.org", "snippet": "Machine learning in Python"},
            {"title": "TensorFlow", "url": "https://tensorflow.org", "snippet": "Open source ML platform"},
            {"title": "PyTorch", "url": "https://pytorch.org", "snippet": "Deep learning framework"}
        ],
        "langchain": [
            {"title": "LangChain Docs", "url": "https://langchain.readthedocs.io", "snippet": "Build LLM applications"},
            {"title": "LangChain GitHub", "url": "https://github.com/langchain-ai/langchain", "snippet": "Source code repository"},
            {"title": "LangChain Examples", "url": "https://langchain.com/examples", "snippet": "Sample implementations"}
        ]
    }
    
    # Find relevant results
    query_lower = query.lower()
    results = []
    
    for key, data in mock_results.items():
        if key in query_lower:
            results = data
            break
    
    # Generate generic results if no match
    if not results:
        results = [
            {"title": f"Result 1 for {query}", "url": f"https://example.com/1", "snippet": f"Information about {query}"},
            {"title": f"Result 2 for {query}", "url": f"https://example.com/2", "snippet": f"More details on {query}"},
            {"title": f"Result 3 for {query}", "url": f"https://example.com/3", "snippet": f"Additional {query} resources"}
        ]
    
    # Limit results
    results = results[:min(num_results, len(results))]
    
    return {
        "query": query,
        "num_results": len(results),
        "results": results,
        "search_time": "0.25 seconds",
        "source": "mock_search_engine"
    }


@tool
def knowledge_base_query(topic: str, category: str = "general") -> Dict[str, Any]:
    """
    Query a mock knowledge base for information on various topics.
    
    Args:
        topic: Topic to search for
        category: Category of information (general, technical, historical)
        
    Returns:
        Dictionary with knowledge base information
        
    Example:
        knowledge_base_query("artificial intelligence", "technical") → AI information
    """
    knowledge_base = {
        "artificial intelligence": {
            "general": "AI refers to computer systems that can perform tasks requiring human intelligence.",
            "technical": "AI includes machine learning, neural networks, and natural language processing.",
            "historical": "AI was founded as an academic discipline in 1956 at Dartmouth College."
        },
        "python": {
            "general": "Python is a high-level programming language known for its simplicity.",
            "technical": "Python uses dynamic typing and automatic memory management.",
            "historical": "Python was created by Guido van Rossum and first released in 1991."
        },
        "machine learning": {
            "general": "ML enables computers to learn and improve from experience automatically.",
            "technical": "ML uses algorithms like supervised, unsupervised, and reinforcement learning.",
            "historical": "The term 'machine learning' was coined by Arthur Samuel in 1959."
        }
    }
    
    topic_lower = topic.lower()
    if topic_lower in knowledge_base:
        info = knowledge_base[topic_lower].get(category, "No information available for this category.")
    else:
        info = f"No specific information found for '{topic}' in category '{category}'."
    
    return {
        "topic": topic,
        "category": category,
        "information": info,
        "source": "internal_knowledge_base",
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# 3. Hybrid Tools (Math + Research)
# =============================================================================

class HybridMathResearchTool(BaseTool):
    """
    Advanced tool that combines mathematical operations with research capabilities.
    Can perform calculations and then search for related information.
    """
    
    name: str = "hybrid_math_research"
    description: str = """
    Perform mathematical calculations and research related information.
    Can calculate statistics and then find relevant research or explanations.
    """
    
    def _run(self, numbers: List[float], research_topic: str = "") -> Dict[str, Any]:
        """
        Execute hybrid math and research operation.
        
        Args:
            numbers: List of numbers for mathematical analysis
            research_topic: Optional topic to research related to the math
            
        Returns:
            Combined mathematical and research results
        """
        # Perform mathematical analysis
        math_result = statistical_analysis.invoke({"numbers": numbers})
        
        # Perform research if topic provided
        research_result = None
        if research_topic:
            research_result = web_search_mock.invoke({
                "query": research_topic,
                "num_results": 2
            })
        
        return {
            "tool_type": "hybrid_math_research",
            "mathematical_analysis": math_result,
            "research_results": research_result,
            "correlation": self._find_correlation(math_result, research_topic) if research_topic else None
        }
    
    def _find_correlation(self, math_result: Dict, research_topic: str) -> str:
        """Find correlation between math results and research topic."""
        if "error" in math_result:
            return "Cannot correlate due to math error"
        
        mean_val = math_result.get("mean", 0)
        
        correlations = {
            "temperature": f"Average value {mean_val:.2f} could represent temperature measurements",
            "stock": f"Mean of {mean_val:.2f} might indicate stock price analysis",
            "performance": f"Average {mean_val:.2f} could be performance metrics",
            "data": f"Statistical mean {mean_val:.2f} represents central tendency in the dataset"
        }
        
        for key, correlation in correlations.items():
            if key in research_topic.lower():
                return correlation
        
        return f"Mathematical mean {mean_val:.2f} provides quantitative insight for {research_topic}"


# =============================================================================
# 4. Specialized Tool Collections
# =============================================================================

class FinancialToolkit:
    """Collection of financial analysis tools."""
    
    @staticmethod
    @tool
    def compound_interest(principal: float, rate: float, time: float, compounds_per_year: int = 1) -> Dict[str, Any]:
        """
        Calculate compound interest.
        
        Args:
            principal: Initial amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            time: Time in years
            compounds_per_year: Number of times interest compounds per year
            
        Returns:
            Dictionary with compound interest calculation results
        """
        try:
            amount = principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)
            interest_earned = amount - principal
            
            return {
                "principal": principal,
                "rate": rate,
                "time_years": time,
                "compounds_per_year": compounds_per_year,
                "final_amount": round(amount, 2),
                "interest_earned": round(interest_earned, 2),
                "total_return_percentage": round((interest_earned / principal) * 100, 2)
            }
        except Exception as e:
            return {"error": f"Compound interest calculation failed: {str(e)}"}
    
    @staticmethod
    @tool
    def loan_payment(principal: float, rate: float, periods: int) -> Dict[str, Any]:
        """
        Calculate monthly loan payment using standard amortization formula.
        
        Args:
            principal: Loan amount
            rate: Monthly interest rate (annual rate / 12)
            periods: Number of payment periods
            
        Returns:
            Dictionary with loan payment information
        """
        try:
            if rate == 0:
                monthly_payment = principal / periods
            else:
                monthly_payment = principal * (rate * (1 + rate)**periods) / ((1 + rate)**periods - 1)
            
            total_paid = monthly_payment * periods
            total_interest = total_paid - principal
            
            return {
                "loan_amount": principal,
                "monthly_rate": rate,
                "periods": periods,
                "monthly_payment": round(monthly_payment, 2),
                "total_amount_paid": round(total_paid, 2),
                "total_interest": round(total_interest, 2),
                "interest_percentage": round((total_interest / principal) * 100, 2)
            }
        except Exception as e:
            return {"error": f"Loan calculation failed: {str(e)}"}


class DataAnalysisToolkit:
    """Collection of data analysis and processing tools."""
    
    @staticmethod
    @tool
    def data_summary(data: List[float], include_distribution: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive data summary with optional distribution analysis.
        
        Args:
            data: List of numerical data points
            include_distribution: Whether to include distribution analysis
            
        Returns:
            Dictionary with complete data summary
        """
        if not data:
            return {"error": "No data provided"}
        
        try:
            np_data = np.array(data)
            
            summary = {
                "basic_stats": {
                    "count": len(data),
                    "mean": float(np.mean(np_data)),
                    "median": float(np.median(np_data)),
                    "std_dev": float(np.std(np_data)),
                    "variance": float(np.var(np_data)),
                    "min": float(np.min(np_data)),
                    "max": float(np.max(np_data)),
                    "range": float(np.max(np_data) - np.min(np_data))
                },
                "percentiles": {
                    "25th": float(np.percentile(np_data, 25)),
                    "50th": float(np.percentile(np_data, 50)),
                    "75th": float(np.percentile(np_data, 75)),
                    "90th": float(np.percentile(np_data, 90)),
                    "95th": float(np.percentile(np_data, 95))
                }
            }
            
            if include_distribution:
                # Calculate skewness and kurtosis approximations
                mean = summary["basic_stats"]["mean"]
                std = summary["basic_stats"]["std_dev"]
                
                # Simple skewness approximation
                skewness_approx = np.mean(((np_data - mean) / std) ** 3) if std > 0 else 0
                
                summary["distribution"] = {
                    "skewness_approx": float(skewness_approx),
                    "is_normal_like": abs(skewness_approx) < 0.5,
                    "outlier_threshold_lower": mean - 2 * std,
                    "outlier_threshold_upper": mean + 2 * std,
                    "potential_outliers": [x for x in data if x < mean - 2*std or x > mean + 2*std]
                }
            
            return summary
            
        except Exception as e:
            return {"error": f"Data analysis failed: {str(e)}"}


# =============================================================================
# 5. Tool Integration and Management
# =============================================================================

class CustomToolManager:
    """
    Manager class for organizing and using custom tools effectively.
    Provides tool discovery, validation, and execution capabilities.
    """
    
    def __init__(self):
        self.tools = {
            # Math tools
            "add_numbers": add_numbers,
            "multiply_numbers": multiply_numbers,
            "statistical_analysis": statistical_analysis,
            "matrix_operations": matrix_operations,
            
            # Research tools
            "web_search": web_search_mock,
            "knowledge_query": knowledge_base_query,
            
            # Hybrid tools
            "hybrid_math_research": HybridMathResearchTool(),
            
            # Financial tools
            "compound_interest": FinancialToolkit.compound_interest,
            "loan_payment": FinancialToolkit.loan_payment,
            
            # Data analysis tools
            "data_summary": DataAnalysisToolkit.data_summary
        }
        
        self.usage_history = []
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools with descriptions."""
        tool_descriptions = {}
        for name, tool in self.tools.items():
            if hasattr(tool, 'description'):
                tool_descriptions[name] = tool.description
            elif hasattr(tool, '__doc__'):
                tool_descriptions[name] = tool.__doc__.split('\n')[0] if tool.__doc__ else "No description"
            else:
                tool_descriptions[name] = "Custom tool"
        
        return tool_descriptions
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result with metadata
        """
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            tool = self.tools[tool_name]
            
            # Record usage
            usage_record = {
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                "parameters": kwargs
            }
            
            # Execute tool
            if hasattr(tool, 'invoke'):
                result = tool.invoke(kwargs)
            elif hasattr(tool, '_run'):
                result = tool._run(**kwargs)
            else:
                result = tool(**kwargs)
            
            usage_record["result"] = result
            self.usage_history.append(usage_record)
            
            return {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_record = {
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                "parameters": kwargs,
                "error": str(e)
            }
            self.usage_history.append(error_record)
            
            return {
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        if not self.usage_history:
            return {"message": "No tool usage recorded yet"}
        
        tool_counts = {}
        error_counts = {}
        
        for record in self.usage_history:
            tool_name = record["tool"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            if "error" in record:
                error_counts[tool_name] = error_counts.get(tool_name, 0) + 1
        
        return {
            "total_executions": len(self.usage_history),
            "tool_usage_counts": tool_counts,
            "error_counts": error_counts,
            "most_used_tool": max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None,
            "success_rate": (len(self.usage_history) - sum(error_counts.values())) / len(self.usage_history) * 100
        }


# =============================================================================
# 6. Demonstration Functions
# =============================================================================

def demonstrate_math_toolkit():
    """Demonstrate the comprehensive math toolkit."""
    print("🧮 Advanced Math Toolkit Demonstration")
    print("=" * 50)
    
    test_numbers = [1, 2, 3, 4, 5, 10, 15, 20]
    
    # Addition with analysis
    print("\n➕ Addition Analysis:")
    add_result = add_numbers.invoke({"numbers": test_numbers})
    print(f"   Numbers: {test_numbers}")
    print(f"   Sum: {add_result.get('sum')}")
    print(f"   Average: {add_result.get('average')}")
    
    # Statistical analysis
    print("\n📊 Statistical Analysis:")
    stats_result = statistical_analysis.invoke({"numbers": test_numbers})
    print(f"   Mean: {stats_result.get('mean'):.2f}")
    print(f"   Median: {stats_result.get('median')}")
    print(f"   Std Dev: {stats_result.get('std_deviation'):.2f}")
    
    # Matrix operations
    print("\n🔢 Matrix Operations:")
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    matrix_result = matrix_operations.invoke({
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "operation": "multiply"
    })
    print(f"   Matrix A: {matrix_a}")
    print(f"   Matrix B: {matrix_b}")
    print(f"   A × B: {matrix_result.get('result')}")


def demonstrate_hybrid_tools():
    """Demonstrate hybrid math and research tools."""
    print("\n🔬 Hybrid Tools Demonstration")
    print("=" * 50)
    
    hybrid_tool = HybridMathResearchTool()
    
    # Test with temperature data
    temperature_data = [68, 72, 75, 71, 69, 73, 76]
    result = hybrid_tool._run(
        numbers=temperature_data,
        research_topic="temperature measurement"
    )
    
    print(f"\n🌡️  Temperature Analysis:")
    print(f"   Data: {temperature_data}")
    
    math_analysis = result.get("mathematical_analysis", {})
    print(f"   Average: {math_analysis.get('mean', 0):.1f}°F")
    print(f"   Range: {math_analysis.get('range', 0):.1f}°F")
    
    correlation = result.get("correlation")
    if correlation:
        print(f"   Insight: {correlation}")


def demonstrate_financial_tools():
    """Demonstrate financial analysis tools."""
    print("\n💰 Financial Tools Demonstration")
    print("=" * 50)
    
    # Compound interest calculation
    print("\n📈 Compound Interest:")
    interest_result = FinancialToolkit.compound_interest.invoke({
        "principal": 10000,
        "rate": 0.07,
        "time": 10,
        "compounds_per_year": 12
    })
    
    print(f"   Principal: ${interest_result.get('principal'):,}")
    print(f"   Rate: {interest_result.get('rate')*100}% annually")
    print(f"   Time: {interest_result.get('time_years')} years")
    print(f"   Final Amount: ${interest_result.get('final_amount'):,}")
    print(f"   Interest Earned: ${interest_result.get('interest_earned'):,}")
    
    # Loan payment calculation
    print("\n🏠 Loan Payment:")
    loan_result = FinancialToolkit.loan_payment.invoke({
        "principal": 300000,
        "rate": 0.05/12,  # 5% annual rate, monthly
        "periods": 30*12  # 30 years
    })
    
    print(f"   Loan Amount: ${loan_result.get('loan_amount'):,}")
    print(f"   Monthly Payment: ${loan_result.get('monthly_payment'):,}")
    print(f"   Total Interest: ${loan_result.get('total_interest'):,}")


def demonstrate_tool_manager():
    """Demonstrate the custom tool manager."""
    print("\n🛠️  Tool Manager Demonstration")
    print("=" * 50)
    
    manager = CustomToolManager()
    
    # Show available tools
    print("\n📋 Available Tools:")
    tools = manager.get_available_tools()
    for name, desc in list(tools.items())[:5]:  # Show first 5
        print(f"   • {name}: {desc[:50]}...")
    
    # Execute some tools
    print("\n🔧 Tool Executions:")
    
    # Math tool
    result1 = manager.execute_tool("add_numbers", numbers=[1, 2, 3, 4, 5])
    print(f"   Addition result: {result1.get('result', {}).get('sum')}")
    
    # Research tool
    result2 = manager.execute_tool("web_search", query="python programming", num_results=2)
    print(f"   Search results: {len(result2.get('result', {}).get('results', []))} found")
    
    # Usage statistics
    print(f"\n📊 Usage Statistics:")
    stats = manager.get_usage_stats()
    print(f"   Total executions: {stats.get('total_executions')}")
    print(f"   Success rate: {stats.get('success_rate'):.1f}%")


# Main demonstration
if __name__ == "__main__":
    print("🎓 Module 3: Building Custom Tools")
    print("=" * 60)
    
    # Demonstrate all tool categories
    demonstrate_math_toolkit()
    demonstrate_hybrid_tools()
    demonstrate_financial_tools()
    demonstrate_tool_manager()
    
    print("\n✅ Module 3 demonstrations completed!")
    print("Next: Module 4 - Agent Architectures")
