"""
Real-World Financial Agent Example
=================================

This example demonstrates how to build a comprehensive financial analysis agent
that can handle stock research, portfolio analysis, and risk calculations.
"""

import os
import json
import requests
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI


# =============================================================================
# Financial Analysis Tools
# =============================================================================

@tool
def get_stock_price(symbol: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Get current stock price and basic information.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "GOOGL")
        period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with stock price information
    """
    try:
        # Use yfinance for real stock data
        stock = yf.Ticker(symbol)
        
        # Get current info
        info = stock.info
        history = stock.history(period=period)
        
        if history.empty:
            return {"error": f"No data found for symbol {symbol}"}
        
        current_price = history['Close'][-1]
        previous_price = history['Close'][-2] if len(history) > 1 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(float(current_price), 2),
            "previous_close": round(float(previous_price), 2),
            "change": round(float(change), 2),
            "change_percent": round(float(change_percent), 2),
            "volume": int(history['Volume'][-1]) if 'Volume' in history else 0,
            "market_cap": info.get('marketCap', 'N/A'),
            "company_name": info.get('longName', symbol.upper()),
            "sector": info.get('sector', 'N/A'),
            "period_analyzed": period,
            "data_points": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Fallback to mock data for demonstration
        return {
            "symbol": symbol.upper(),
            "current_price": 150.00,
            "previous_close": 148.50,
            "change": 1.50,
            "change_percent": 1.01,
            "volume": 1000000,
            "market_cap": "2.5T",
            "company_name": f"{symbol.upper()} Corporation",
            "sector": "Technology",
            "period_analyzed": period,
            "data_points": 30,
            "timestamp": datetime.now().isoformat(),
            "note": f"Mock data (yfinance error: {str(e)})"
        }


@tool
def calculate_portfolio_performance(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        holdings: List of holdings with format [{"symbol": "AAPL", "shares": 100, "purchase_price": 140.0}, ...]
        
    Returns:
        Dictionary with portfolio analysis
    """
    if not holdings:
        return {"error": "No holdings provided"}
    
    try:
        portfolio_analysis = {
            "total_holdings": len(holdings),
            "positions": [],
            "total_investment": 0,
            "current_value": 0,
            "total_gain_loss": 0,
            "total_return_percent": 0
        }
        
        for holding in holdings:
            symbol = holding.get("symbol", "")
            shares = holding.get("shares", 0)
            purchase_price = holding.get("purchase_price", 0)
            
            # Get current price
            stock_data = get_stock_price.invoke({"symbol": symbol})
            
            if "error" not in stock_data:
                current_price = stock_data["current_price"]
                investment = shares * purchase_price
                current_value = shares * current_price
                gain_loss = current_value - investment
                return_percent = (gain_loss / investment) * 100 if investment > 0 else 0
                
                position = {
                    "symbol": symbol,
                    "shares": shares,
                    "purchase_price": purchase_price,
                    "current_price": current_price,
                    "investment": round(investment, 2),
                    "current_value": round(current_value, 2),
                    "gain_loss": round(gain_loss, 2),
                    "return_percent": round(return_percent, 2)
                }
                
                portfolio_analysis["positions"].append(position)
                portfolio_analysis["total_investment"] += investment
                portfolio_analysis["current_value"] += current_value
        
        # Calculate totals
        total_gain_loss = portfolio_analysis["current_value"] - portfolio_analysis["total_investment"]
        total_return_percent = (total_gain_loss / portfolio_analysis["total_investment"]) * 100 if portfolio_analysis["total_investment"] > 0 else 0
        
        portfolio_analysis["total_gain_loss"] = round(total_gain_loss, 2)
        portfolio_analysis["total_return_percent"] = round(total_return_percent, 2)
        portfolio_analysis["total_investment"] = round(portfolio_analysis["total_investment"], 2)
        portfolio_analysis["current_value"] = round(portfolio_analysis["current_value"], 2)
        
        # Add performance metrics
        portfolio_analysis["performance_summary"] = {
            "best_performer": max(portfolio_analysis["positions"], key=lambda x: x["return_percent"])["symbol"] if portfolio_analysis["positions"] else "None",
            "worst_performer": min(portfolio_analysis["positions"], key=lambda x: x["return_percent"])["symbol"] if portfolio_analysis["positions"] else "None",
            "average_return": round(sum(pos["return_percent"] for pos in portfolio_analysis["positions"]) / len(portfolio_analysis["positions"]), 2) if portfolio_analysis["positions"] else 0
        }
        
        return portfolio_analysis
        
    except Exception as e:
        return {"error": f"Portfolio calculation failed: {str(e)}"}


@tool
def risk_assessment(symbols: List[str], investment_amount: float) -> Dict[str, Any]:
    """
    Perform risk assessment for given stocks.
    
    Args:
        symbols: List of stock symbols to analyze
        investment_amount: Total investment amount to assess risk for
        
    Returns:
        Dictionary with risk analysis
    """
    if not symbols:
        return {"error": "No symbols provided"}
    
    try:
        risk_analysis = {
            "symbols_analyzed": symbols,
            "investment_amount": investment_amount,
            "individual_risks": [],
            "portfolio_risk": {},
            "recommendations": []
        }
        
        volatilities = []
        sector_diversification = set()
        
        for symbol in symbols:
            # Get stock data
            stock_data = get_stock_price.invoke({"symbol": symbol, "period": "3mo"})
            
            if "error" not in stock_data:
                # Simple volatility calculation (using change percent as proxy)
                volatility = abs(stock_data.get("change_percent", 0))
                
                # Risk categories
                if volatility < 2:
                    risk_level = "Low"
                elif volatility < 5:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                individual_risk = {
                    "symbol": symbol,
                    "current_price": stock_data.get("current_price", 0),
                    "volatility_proxy": round(volatility, 2),
                    "risk_level": risk_level,
                    "sector": stock_data.get("sector", "Unknown"),
                    "market_cap": stock_data.get("market_cap", "N/A")
                }
                
                risk_analysis["individual_risks"].append(individual_risk)
                volatilities.append(volatility)
                if stock_data.get("sector") != "N/A":
                    sector_diversification.add(stock_data.get("sector"))
        
        # Portfolio-level risk assessment
        if volatilities:
            avg_volatility = sum(volatilities) / len(volatilities)
            max_volatility = max(volatilities)
            
            # Portfolio risk level
            if avg_volatility < 3:
                portfolio_risk_level = "Conservative"
            elif avg_volatility < 6:
                portfolio_risk_level = "Moderate"
            else:
                portfolio_risk_level = "Aggressive"
            
            risk_analysis["portfolio_risk"] = {
                "average_volatility": round(avg_volatility, 2),
                "maximum_volatility": round(max_volatility, 2),
                "risk_level": portfolio_risk_level,
                "diversification_score": len(sector_diversification),
                "sectors_represented": list(sector_diversification)
            }
            
            # Generate recommendations
            recommendations = []
            
            if len(sector_diversification) < 3:
                recommendations.append("Consider diversifying across more sectors to reduce risk")
            
            if avg_volatility > 7:
                recommendations.append("Portfolio shows high volatility - consider adding some stable assets")
            
            if investment_amount > 100000 and len(symbols) < 5:
                recommendations.append("For large investments, consider spreading across more positions")
            
            if max_volatility > 10:
                recommendations.append("Some positions show very high volatility - monitor closely")
            
            if not recommendations:
                recommendations.append("Portfolio shows reasonable risk distribution")
            
            risk_analysis["recommendations"] = recommendations
        
        return risk_analysis
        
    except Exception as e:
        return {"error": f"Risk assessment failed: {str(e)}"}


@tool
def market_sentiment_analyzer(symbols: List[str]) -> Dict[str, Any]:
    """
    Analyze market sentiment for given stocks (simplified implementation).
    
    Args:
        symbols: List of stock symbols to analyze sentiment for
        
    Returns:
        Dictionary with sentiment analysis
    """
    try:
        sentiment_analysis = {
            "symbols_analyzed": symbols,
            "sentiment_scores": [],
            "overall_sentiment": "",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        sentiments = []
        
        for symbol in symbols:
            # Get recent stock performance as sentiment proxy
            stock_data = get_stock_price.invoke({"symbol": symbol, "period": "1mo"})
            
            if "error" not in stock_data:
                change_percent = stock_data.get("change_percent", 0)
                
                # Simple sentiment scoring based on recent performance
                if change_percent > 5:
                    sentiment = "Very Positive"
                    score = 0.8
                elif change_percent > 2:
                    sentiment = "Positive"
                    score = 0.6
                elif change_percent > -2:
                    sentiment = "Neutral"
                    score = 0.5
                elif change_percent > -5:
                    sentiment = "Negative"
                    score = 0.3
                else:
                    sentiment = "Very Negative"
                    score = 0.1
                
                sentiment_data = {
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "confidence_score": score,
                    "recent_performance": f"{change_percent:+.2f}%",
                    "volume_indicator": "High" if stock_data.get("volume", 0) > 1000000 else "Normal"
                }
                
                sentiment_analysis["sentiment_scores"].append(sentiment_data)
                sentiments.append(score)
        
        # Overall sentiment
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            if avg_sentiment > 0.7:
                overall = "Bullish"
            elif avg_sentiment > 0.5:
                overall = "Neutral to Positive"
            elif avg_sentiment > 0.3:
                overall = "Neutral to Negative"
            else:
                overall = "Bearish"
            
            sentiment_analysis["overall_sentiment"] = overall
            sentiment_analysis["average_confidence"] = round(avg_sentiment, 2)
        
        return sentiment_analysis
        
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}


# =============================================================================
# Financial Agent Implementation
# =============================================================================

class FinancialAgent:
    """Comprehensive financial analysis agent."""
    
    def __init__(self, llm=None):
        self.tools = [
            get_stock_price,
            calculate_portfolio_performance,
            risk_assessment,
            market_sentiment_analyzer
        ]
        
        if llm:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5
            )
            self.has_llm = True
        else:
            self.agent = None
            self.has_llm = False
    
    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive stock analysis."""
        try:
            # Get basic stock data
            stock_data = get_stock_price.invoke({"symbol": symbol})
            
            # Perform risk assessment
            risk_data = risk_assessment.invoke({
                "symbols": [symbol],
                "investment_amount": 10000
            })
            
            # Get sentiment
            sentiment_data = market_sentiment_analyzer.invoke({"symbols": [symbol]})
            
            return {
                "symbol": symbol,
                "stock_data": stock_data,
                "risk_analysis": risk_data,
                "sentiment": sentiment_data,
                "analysis_type": "comprehensive_stock_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Stock analysis failed: {str(e)}"}
    
    def analyze_portfolio(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive portfolio analysis."""
        try:
            # Calculate performance
            performance = calculate_portfolio_performance.invoke({"holdings": holdings})
            
            # Extract symbols for risk and sentiment analysis
            symbols = [holding["symbol"] for holding in holdings]
            
            # Risk assessment
            risk_data = risk_assessment.invoke({
                "symbols": symbols,
                "investment_amount": performance.get("total_investment", 0)
            })
            
            # Sentiment analysis
            sentiment_data = market_sentiment_analyzer.invoke({"symbols": symbols})
            
            return {
                "portfolio_performance": performance,
                "risk_analysis": risk_data,
                "market_sentiment": sentiment_data,
                "analysis_type": "comprehensive_portfolio_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Portfolio analysis failed: {str(e)}"}
    
    def interactive_query(self, query: str) -> str:
        """Handle natural language financial queries."""
        if self.has_llm:
            try:
                return self.agent.run(query)
            except Exception as e:
                return f"Agent error: {str(e)}"
        else:
            # Simple pattern matching for demo
            query_lower = query.lower()
            
            if "price" in query_lower and any(symbol in query.upper() for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]):
                # Extract symbol (simplified)
                for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                    if symbol in query.upper():
                        result = get_stock_price.invoke({"symbol": symbol})
                        return f"{symbol} is currently trading at ${result.get('current_price', 'N/A')} ({result.get('change_percent', 0):+.2f}%)"
            
            elif "portfolio" in query_lower:
                return "To analyze a portfolio, please use the analyze_portfolio() method with your holdings data."
            
            elif "risk" in query_lower:
                return "For risk analysis, please specify the stocks you want to analyze using the risk_assessment tool."
            
            else:
                return f"I understand you're asking about: {query}. Please use specific methods like analyze_stock() or analyze_portfolio() for detailed analysis."


# =============================================================================
# Example Usage and Demonstrations
# =============================================================================

def demonstrate_financial_agent():
    """Demonstrate the financial agent capabilities."""
    print("💰 Financial Agent Demonstration")
    print("=" * 60)
    
    # Initialize agent
    agent = FinancialAgent()
    
    # Example 1: Single stock analysis
    print("\n📊 Single Stock Analysis:")
    print("-" * 30)
    
    stock_analysis = agent.analyze_stock("AAPL")
    if "error" not in stock_analysis:
        stock_data = stock_analysis["stock_data"]
        print(f"Stock: {stock_data.get('company_name', 'Unknown')} ({stock_data.get('symbol', 'N/A')})")
        print(f"Price: ${stock_data.get('current_price', 0)} ({stock_data.get('change_percent', 0):+.2f}%)")
        print(f"Sector: {stock_data.get('sector', 'Unknown')}")
        
        risk_data = stock_analysis["risk_analysis"]
        if "error" not in risk_data and risk_data.get("individual_risks"):
            risk_info = risk_data["individual_risks"][0]
            print(f"Risk Level: {risk_info.get('risk_level', 'Unknown')}")
    else:
        print(f"Error: {stock_analysis['error']}")
    
    # Example 2: Portfolio analysis
    print("\n📈 Portfolio Analysis:")
    print("-" * 30)
    
    sample_portfolio = [
        {"symbol": "AAPL", "shares": 50, "purchase_price": 140.0},
        {"symbol": "GOOGL", "shares": 10, "purchase_price": 2500.0},
        {"symbol": "MSFT", "shares": 25, "purchase_price": 300.0}
    ]
    
    portfolio_analysis = agent.analyze_portfolio(sample_portfolio)
    if "error" not in portfolio_analysis:
        performance = portfolio_analysis["portfolio_performance"]
        print(f"Total Investment: ${performance.get('total_investment', 0):,.2f}")
        print(f"Current Value: ${performance.get('current_value', 0):,.2f}")
        print(f"Total Return: {performance.get('total_return_percent', 0):+.2f}%")
        
        if "performance_summary" in performance:
            summary = performance["performance_summary"]
            print(f"Best Performer: {summary.get('best_performer', 'N/A')}")
            print(f"Average Return: {summary.get('average_return', 0):.2f}%")
    else:
        print(f"Error: {portfolio_analysis['error']}")
    
    # Example 3: Interactive queries
    print("\n💬 Interactive Queries:")
    print("-" * 30)
    
    sample_queries = [
        "What's the current price of AAPL?",
        "How is my portfolio performing?",
        "What's the risk level of TSLA?"
    ]
    
    for query in sample_queries:
        print(f"\nQuery: {query}")
        response = agent.interactive_query(query)
        print(f"Response: {response}")


# =============================================================================
# Advanced Features
# =============================================================================

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    Convert currency amounts (mock implementation).
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
        
    Returns:
        Dictionary with conversion result
    """
    # Mock exchange rates for demonstration
    exchange_rates = {
        ("USD", "EUR"): 0.85,
        ("USD", "GBP"): 0.73,
        ("USD", "JPY"): 110.0,
        ("EUR", "USD"): 1.18,
        ("GBP", "USD"): 1.37,
        ("JPY", "USD"): 0.009
    }
    
    rate_key = (from_currency.upper(), to_currency.upper())
    
    if rate_key in exchange_rates:
        rate = exchange_rates[rate_key]
        converted_amount = amount * rate
        
        return {
            "original_amount": amount,
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "exchange_rate": rate,
            "converted_amount": round(converted_amount, 2),
            "timestamp": datetime.now().isoformat(),
            "source": "mock_exchange_api"
        }
    else:
        return {
            "error": f"Exchange rate not available for {from_currency} to {to_currency}",
            "available_rates": list(exchange_rates.keys())
        }


if __name__ == "__main__":
    print("🏦 Financial Agent Example")
    print("=" * 50)
    
    # Run the demonstration
    demonstrate_financial_agent()
    
    print("\n✅ Financial agent demonstration completed!")
    print("\n💡 Next steps:")
    print("  • Set up real API keys for live data")
    print("  • Integrate with real LLM for natural language processing")
    print("  • Add more sophisticated risk models")
    print("  • Implement real-time alerts and monitoring")
