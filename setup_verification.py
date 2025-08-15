# AI Agents LangChain Course - Setup and Testing

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """Check if all required packages are installed."""
    required_packages = [
        "langchain",
        "langchain_community", 
        "langchain_openai",
        "openai",
        "requests",
        "numpy",
        "pandas",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 To install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🧪 Testing Basic Functionality:")
    
    try:
        # Test LangChain tools
        from langchain.tools import tool
        
        @tool
        def test_tool(input_text: str) -> str:
            """Test tool for verification."""
            return f"Tool received: {input_text}"
        
        result = test_tool.invoke({"input_text": "Hello World"})
        if "Hello World" in result:
            print("✅ LangChain tools working")
        else:
            print("❌ LangChain tools issue")
            return False
            
    except Exception as e:
        print(f"❌ LangChain tools error: {e}")
        return False
    
    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3])
        if arr.mean() == 2.0:
            print("✅ NumPy working")
        else:
            print("❌ NumPy calculation issue")
            return False
            
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False
    
    try:
        # Test course modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test module 1
        from modules.introduction.concepts import AgentCharacteristics
        agent_chars = AgentCharacteristics(
            has_tools=True,
            maintains_memory=True, 
            can_plan=True,
            accesses_real_time_data=True,
            is_stateful=True
        )
        print("✅ Module 1 concepts working")
        
        # Test module 2  
        from modules.tool_calling.tool_calling_basics import simple_calculator
        calc_result = simple_calculator.invoke({"expression": "2+2"})
        if "4" in calc_result:
            print("✅ Module 2 tools working")
        else:
            print("❌ Module 2 calculation issue")
            return False
            
    except Exception as e:
        print(f"❌ Course modules error: {e}")
        return False
    
    return True

def run_example_tests():
    """Run tests on example applications."""
    print("\n🎯 Testing Example Applications:")
    
    try:
        # Test financial agent
        from examples.financial_agent import FinancialAgent, get_stock_price
        
        # Test stock price tool
        stock_data = get_stock_price.invoke({"symbol": "AAPL"})
        if "current_price" in stock_data:
            print("✅ Financial agent tools working")
        else:
            print("❌ Financial agent tools issue")
            return False
        
        # Test agent initialization
        agent = FinancialAgent()
        if agent:
            print("✅ Financial agent initialization working")
        else:
            print("❌ Financial agent initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Financial agent error: {e}")
        return False
    
    try:
        # Test utilities
        from utils.agent_utils import AgentMonitor, AgentConfig
        
        monitor = AgentMonitor()
        config = AgentConfig()
        
        if monitor and config:
            print("✅ Agent utilities working")
        else:
            print("❌ Agent utilities issue")
            return False
            
    except Exception as e:
        print(f"❌ Agent utilities error: {e}")
        return False
    
    return True

def check_environment_setup():
    """Check environment configuration."""
    print("\n🌍 Environment Setup:")
    
    # Check for .env file
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file found")
        
        # Check for key variables
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "your_openai_api_key_here":
            print("✅ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not configured (will use mock implementations)")
            
    else:
        print("⚠️  .env file not found (copy from .env.example)")
    
    # Check Jupyter availability
    try:
        import jupyter
        print("✅ Jupyter available for notebooks")
    except ImportError:
        print("⚠️  Jupyter not installed (optional for notebooks)")
    
    return True

def main():
    """Run complete setup verification."""
    print("🚀 AI Agents LangChain Course - Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📦 Package Installation Check:")
    if not check_required_packages():
        all_good = False
    
    # Test functionality
    if not test_basic_functionality():
        all_good = False
    
    # Test examples
    if not run_example_tests():
        all_good = False
    
    # Check environment
    if not check_environment_setup():
        all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Setup verification PASSED! Ready to start the course.")
        print("\n📚 Next steps:")
        print("  1. Open 'notebooks/00_course_introduction.ipynb' to start learning")
        print("  2. Work through the modules sequentially")
        print("  3. Run the example applications")
        print("  4. Build your own agents!")
    else:
        print("❌ Setup verification FAILED. Please fix the issues above.")
        print("\n🔧 Common solutions:")
        print("  • Run: pip install -r requirements.txt")
        print("  • Copy .env.example to .env and add your API keys")
        print("  • Make sure Python 3.8+ is installed")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
