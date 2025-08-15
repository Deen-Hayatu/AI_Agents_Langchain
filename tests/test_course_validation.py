"""
Test suite for AI Agents LangChain Course
Educational validation and module testing
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_module_imports():
    """Test that all course modules can be imported successfully"""
    try:
        from modules.introduction import concepts
        assert hasattr(concepts, 'get_course_overview'), "Course overview function missing"
        print("✅ Introduction module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import introduction module: {e}")

def test_tool_calling_module():
    """Test tool calling basics module"""
    try:
        from modules.tool_calling import tool_calling_basics
        assert hasattr(tool_calling_basics, 'create_basic_tool'), "Basic tool function missing"
        print("✅ Tool calling module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import tool calling module: {e}")

def test_utils_import():
    """Test utility functions"""
    try:
        from utils.agent_utils import AgentMonitor
        assert AgentMonitor, "AgentMonitor class missing"
        print("✅ Agent utils imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import agent utils: {e}")

def test_examples_structure():
    """Test that examples directory has expected structure"""
    examples_dir = project_root / "examples"
    assert examples_dir.exists(), "Examples directory missing"
    
    financial_agent = examples_dir / "financial_agent.py"
    assert financial_agent.exists(), "Financial agent example missing"
    print("✅ Examples structure validated")

def test_requirements_dependencies():
    """Test that key dependencies are available"""
    try:
        import langchain
        import numpy
        import pandas
        print("✅ Core dependencies available")
    except ImportError as e:
        pytest.fail(f"Missing required dependency: {e}")

if __name__ == "__main__":
    # Run tests directly
    test_module_imports()
    test_tool_calling_module()
    test_utils_import()
    test_examples_structure()
    test_requirements_dependencies()
    print("🎉 All educational validation tests passed!")
