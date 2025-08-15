<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for AI Agents LangChain Course

## Project Context
This is an educational project focused on teaching AI agents development using LangChain and LangGraph. The course covers practical implementations from basic concepts to advanced real-world applications.

## Code Style Guidelines
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and returns
- Include comprehensive docstrings for all classes and functions
- Prefer explicit imports over wildcard imports
- Use descriptive variable and function names

## LangChain Specific Guidelines
- Always use the latest LangChain patterns and best practices
- Prefer LangGraph over legacy initialize_agent for complex workflows
- Use proper tool schemas with @tool decorator
- Include proper error handling for tool calls
- Use structured outputs when possible

## Educational Focus
- Include clear comments explaining complex concepts
- Provide practical examples that build progressively
- Add error handling with educational explanations
- Include debugging tips and common pitfalls
- Make code modular and reusable for learning purposes

## Documentation Requirements
- Include usage examples in docstrings
- Add inline comments for complex logic
- Provide clear parameter descriptions
- Include return value explanations
- Add references to relevant LangChain documentation

## Testing and Quality
- Write unit tests for custom tools and utilities
- Include integration tests for agent workflows
- Add validation for tool inputs and outputs
- Handle API rate limits and errors gracefully
- Include performance considerations in complex agents
