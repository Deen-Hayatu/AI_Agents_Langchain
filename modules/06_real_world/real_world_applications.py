"""
Module 6: Real-World Applications
=================================

This module demonstrates practical, production-ready AI agent implementations
for common business use cases including customer support, financial analysis,
healthcare assistance, and research automation.

Learning Objectives:
- Build customer support agents with FAQ and ticket management
- Create financial analysis agents for trading and risk assessment
- Implement healthcare assistants with symptom checking
- Develop research automation agents
- Learn production deployment patterns
"""

import json
import random
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from langchain.tools import tool


# =============================================================================
# 1. Customer Support Agent System
# =============================================================================

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SupportTicket:
    """Represents a customer support ticket."""
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    category: str
    created_at: datetime
    updated_at: datetime
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None


class CustomerSupportAgent:
    """
    Comprehensive customer support agent with FAQ, ticket management,
    and escalation capabilities.
    """
    
    def __init__(self):
        self.faq_database = self._load_faq_database()
        self.tickets: Dict[str, SupportTicket] = {}
        self.escalation_rules = self._load_escalation_rules()
    
    def _load_faq_database(self) -> Dict[str, Dict[str, str]]:
        """Load FAQ database with common questions and answers."""
        return {
            "account": {
                "reset_password": "To reset your password, go to Settings > Account > Reset Password and follow the instructions.",
                "update_email": "You can update your email in Settings > Profile > Contact Information.",
                "delete_account": "To delete your account, contact our support team or visit Settings > Account > Delete Account."
            },
            "billing": {
                "payment_methods": "We accept credit cards, PayPal, and bank transfers. Go to Billing > Payment Methods to manage.",
                "billing_cycle": "Our billing cycles are monthly or annually. You can change this in Billing > Subscription.",
                "refund_policy": "We offer full refunds within 30 days of purchase. Contact support for refund requests."
            },
            "technical": {
                "app_crashes": "If the app crashes, try restarting it. If problems persist, clear cache or reinstall.",
                "slow_performance": "For slow performance, check your internet connection and close other applications.",
                "login_issues": "For login issues, verify your credentials and check if your account is active."
            },
            "features": {
                "data_export": "You can export your data from Settings > Data > Export. Choose your preferred format.",
                "integrations": "We support integrations with popular tools. Check Settings > Integrations for available options.",
                "mobile_app": "Our mobile app is available on iOS and Android. Download from your device's app store."
            }
        }
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load escalation rules for ticket management."""
        return {
            "priority_escalation": {
                TicketPriority.URGENT: {"escalate_after_minutes": 15, "escalate_to": "senior_agent"},
                TicketPriority.HIGH: {"escalate_after_minutes": 60, "escalate_to": "team_lead"},
                TicketPriority.MEDIUM: {"escalate_after_minutes": 240, "escalate_to": "supervisor"},
                TicketPriority.LOW: {"escalate_after_minutes": 1440, "escalate_to": "manager"}
            },
            "category_specialists": {
                "billing": "billing_specialist",
                "technical": "tech_specialist", 
                "account": "account_specialist"
            }
        }
    
    @tool
    def search_faq(self, query: str, category: str = "all") -> Dict[str, Any]:
        """
        Search FAQ database for answers to common questions.
        
        Args:
            query: Customer's question or keywords
            category: FAQ category to search (account, billing, technical, features, all)
            
        Returns:
            Dictionary with matching FAQ entries and relevance scores
        """
        query_lower = query.lower()
        results = []
        
        categories_to_search = [category] if category != "all" else self.faq_database.keys()
        
        for cat in categories_to_search:
            if cat in self.faq_database:
                for question, answer in self.faq_database[cat].items():
                    # Simple relevance scoring based on keyword matching
                    relevance = 0
                    question_words = question.replace("_", " ").split()
                    
                    for word in query_lower.split():
                        if word in question.lower() or word in answer.lower():
                            relevance += 1
                    
                    if relevance > 0:
                        results.append({
                            "category": cat,
                            "question": question.replace("_", " ").title(),
                            "answer": answer,
                            "relevance_score": relevance
                        })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": query,
            "category_searched": category,
            "results_found": len(results),
            "results": results[:5],  # Top 5 results
            "search_timestamp": datetime.now().isoformat()
        }
    
    @tool
    def create_support_ticket(self, customer_id: str, subject: str, description: str, 
                            category: str = "general") -> Dict[str, Any]:
        """
        Create a new support ticket for complex issues.
        
        Args:
            customer_id: Unique identifier for the customer
            subject: Brief description of the issue
            description: Detailed description of the problem
            category: Issue category (account, billing, technical, features)
            
        Returns:
            Dictionary with ticket information and next steps
        """
        # Generate ticket ID
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        
        # Determine priority based on keywords
        priority = self._determine_priority(subject, description)
        
        # Create ticket
        ticket = SupportTicket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=priority,
            status=TicketStatus.OPEN,
            category=category,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store ticket
        self.tickets[ticket_id] = ticket
        
        # Determine assignment based on category and priority
        assigned_agent = self._assign_agent(ticket)
        ticket.assigned_agent = assigned_agent
        
        return {
            "ticket_id": ticket_id,
            "priority": priority.value,
            "status": ticket.status.value,
            "assigned_agent": assigned_agent,
            "estimated_response_time": self._get_response_time(priority),
            "category": category,
            "created_at": ticket.created_at.isoformat(),
            "next_steps": self._get_next_steps(ticket)
        }
    
    def _determine_priority(self, subject: str, description: str) -> TicketPriority:
        """Determine ticket priority based on content analysis."""
        text = (subject + " " + description).lower()
        
        # Urgent keywords
        if any(word in text for word in ["urgent", "critical", "down", "outage", "emergency", "asap"]):
            return TicketPriority.URGENT
        
        # High priority keywords
        elif any(word in text for word in ["important", "billing issue", "payment", "security", "hack"]):
            return TicketPriority.HIGH
        
        # Medium priority keywords
        elif any(word in text for word in ["bug", "error", "problem", "issue", "slow"]):
            return TicketPriority.MEDIUM
        
        else:
            return TicketPriority.LOW
    
    def _assign_agent(self, ticket: SupportTicket) -> str:
        """Assign agent based on category and workload."""
        specialists = self.escalation_rules["category_specialists"]
        
        if ticket.category in specialists:
            return specialists[ticket.category]
        else:
            return "general_support_agent"
    
    def _get_response_time(self, priority: TicketPriority) -> str:
        """Get estimated response time based on priority."""
        response_times = {
            TicketPriority.URGENT: "15 minutes",
            TicketPriority.HIGH: "1 hour", 
            TicketPriority.MEDIUM: "4 hours",
            TicketPriority.LOW: "24 hours"
        }
        return response_times[priority]
    
    def _get_next_steps(self, ticket: SupportTicket) -> List[str]:
        """Generate next steps for the customer."""
        return [
            f"Your ticket #{ticket.ticket_id} has been created and assigned to {ticket.assigned_agent}",
            f"Expected response time: {self._get_response_time(ticket.priority)}",
            "You will receive email updates on ticket progress",
            "Reply to this ticket for additional information"
        ]


# =============================================================================
# 2. Financial Analysis Agent System
# =============================================================================

@dataclass
class StockData:
    """Represents stock market data."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    pe_ratio: Optional[float]
    timestamp: datetime


@dataclass
class PortfolioPosition:
    """Represents a position in an investment portfolio."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    gain_loss: float
    gain_loss_percent: float


class FinancialAnalysisAgent:
    """
    Comprehensive financial analysis agent for stock analysis,
    portfolio management, and risk assessment.
    """
    
    def __init__(self):
        self.market_data = self._initialize_market_data()
        self.risk_profiles = self._load_risk_profiles()
    
    def _initialize_market_data(self) -> Dict[str, StockData]:
        """Initialize mock market data for demonstration."""
        stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        market_data = {}
        
        for symbol in stocks:
            # Generate realistic mock data
            base_price = random.uniform(50, 500)
            change = random.uniform(-10, 10)
            
            market_data[symbol] = StockData(
                symbol=symbol,
                price=base_price,
                change=change,
                change_percent=(change / base_price) * 100,
                volume=random.randint(1000000, 50000000),
                market_cap=base_price * random.randint(1000000, 5000000),
                pe_ratio=random.uniform(15, 35) if random.random() > 0.1 else None,
                timestamp=datetime.now()
            )
        
        return market_data
    
    def _load_risk_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load investment risk profiles."""
        return {
            "conservative": {
                "risk_tolerance": "low",
                "max_single_position": 0.05,  # 5% max per stock
                "sector_limits": 0.15,  # 15% max per sector
                "preferred_assets": ["bonds", "dividend_stocks", "blue_chips"],
                "volatility_limit": 0.15
            },
            "moderate": {
                "risk_tolerance": "medium",
                "max_single_position": 0.10,  # 10% max per stock
                "sector_limits": 0.25,  # 25% max per sector 
                "preferred_assets": ["large_cap", "etfs", "some_growth"],
                "volatility_limit": 0.25
            },
            "aggressive": {
                "risk_tolerance": "high",
                "max_single_position": 0.20,  # 20% max per stock
                "sector_limits": 0.40,  # 40% max per sector
                "preferred_assets": ["growth_stocks", "small_cap", "crypto"],
                "volatility_limit": 0.40
            }
        }
    
    @tool
    def get_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock analysis including price, trends, and recommendations.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            
        Returns:
            Dictionary with stock analysis and recommendations
        """
        symbol = symbol.upper()
        
        if symbol not in self.market_data:
            return {"error": f"Stock symbol '{symbol}' not found in database"}
        
        stock = self.market_data[symbol]
        
        # Generate technical analysis
        technical_analysis = self._generate_technical_analysis(stock)
        
        # Generate fundamental analysis
        fundamental_analysis = self._generate_fundamental_analysis(stock)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(stock, technical_analysis, fundamental_analysis)
        
        return {
            "symbol": symbol,
            "current_price": stock.price,
            "daily_change": stock.change,
            "daily_change_percent": round(stock.change_percent, 2),
            "volume": stock.volume,
            "market_cap": stock.market_cap,
            "pe_ratio": stock.pe_ratio,
            "technical_analysis": technical_analysis,
            "fundamental_analysis": fundamental_analysis,
            "recommendation": recommendation,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    @tool
    def analyze_portfolio(self, positions: List[Dict[str, Any]], risk_profile: str = "moderate") -> Dict[str, Any]:
        """
        Analyze investment portfolio for risk, diversification, and performance.
        
        Args:
            positions: List of portfolio positions with symbol, quantity, avg_cost
            risk_profile: Investment risk profile (conservative, moderate, aggressive)
            
        Returns:
            Dictionary with portfolio analysis and recommendations
        """
        if risk_profile not in self.risk_profiles:
            return {"error": f"Unknown risk profile: {risk_profile}"}
        
        profile = self.risk_profiles[risk_profile]
        portfolio_positions = []
        total_value = 0
        
        # Calculate current values for each position
        for pos in positions:
            symbol = pos["symbol"].upper()
            if symbol in self.market_data:
                current_price = self.market_data[symbol].price
                quantity = pos["quantity"]
                avg_cost = pos["avg_cost"]
                
                market_value = quantity * current_price
                cost_basis = quantity * avg_cost
                gain_loss = market_value - cost_basis
                gain_loss_percent = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                portfolio_position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    gain_loss=gain_loss,
                    gain_loss_percent=gain_loss_percent
                )
                
                portfolio_positions.append(portfolio_position)
                total_value += market_value
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_positions, total_value)
        
        # Risk analysis
        risk_analysis = self._analyze_portfolio_risk(portfolio_positions, total_value, profile)
        
        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(portfolio_positions, risk_analysis, profile)
        
        return {
            "portfolio_value": total_value,
            "total_positions": len(portfolio_positions),
            "risk_profile": risk_profile,
            "portfolio_metrics": portfolio_metrics,
            "risk_analysis": risk_analysis,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_technical_analysis(self, stock: StockData) -> Dict[str, Any]:
        """Generate mock technical analysis."""
        # Simulate technical indicators
        return {
            "trend": "bullish" if stock.change > 0 else "bearish",
            "support_level": stock.price * 0.95,
            "resistance_level": stock.price * 1.05,
            "volatility": "high" if abs(stock.change_percent) > 3 else "normal",
            "volume_analysis": "above_average" if stock.volume > 10000000 else "normal"
        }
    
    def _generate_fundamental_analysis(self, stock: StockData) -> Dict[str, Any]:
        """Generate mock fundamental analysis."""
        return {
            "valuation": "fair" if stock.pe_ratio and 15 <= stock.pe_ratio <= 25 else "unknown",
            "market_cap_category": self._categorize_market_cap(stock.market_cap),
            "financial_strength": "strong",  # Mock rating
            "growth_prospects": "positive"  # Mock assessment
        }
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize stock by market capitalization."""
        if market_cap > 200000000000:  # $200B+
            return "mega_cap"
        elif market_cap > 10000000000:  # $10B+
            return "large_cap"
        elif market_cap > 2000000000:   # $2B+
            return "mid_cap"
        else:
            return "small_cap"
    
    def _generate_recommendation(self, stock: StockData, technical: Dict, fundamental: Dict) -> Dict[str, Any]:
        """Generate investment recommendation."""
        score = 0
        
        # Technical scoring
        if technical["trend"] == "bullish":
            score += 1
        if technical["volatility"] == "normal":
            score += 0.5
        
        # Fundamental scoring
        if fundamental["valuation"] == "fair":
            score += 1
        if fundamental["financial_strength"] == "strong":
            score += 0.5
        
        # Generate recommendation
        if score >= 2:
            action = "BUY"
        elif score >= 1:
            action = "HOLD"
        else:
            action = "SELL"
        
        return {
            "action": action,
            "confidence": min(score / 2.5, 1.0),
            "target_price": stock.price * (1.1 if action == "BUY" else 0.95),
            "reasoning": f"Based on {technical['trend']} trend and {fundamental['valuation']} valuation"
        }
    
    def _calculate_portfolio_metrics(self, positions: List[PortfolioPosition], total_value: float) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        total_gain_loss = sum(pos.gain_loss for pos in positions)
        total_cost_basis = total_value - total_gain_loss
        
        return {
            "total_return": total_gain_loss,
            "total_return_percent": (total_gain_loss / total_cost_basis) * 100 if total_cost_basis > 0 else 0,
            "best_performer": max(positions, key=lambda p: p.gain_loss_percent).symbol if positions else None,
            "worst_performer": min(positions, key=lambda p: p.gain_loss_percent).symbol if positions else None,
            "diversification_score": min(len(positions) / 10, 1.0)  # Simple diversification metric
        }
    
    def _analyze_portfolio_risk(self, positions: List[PortfolioPosition], total_value: float, 
                               profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio risk against risk profile."""
        # Check position concentration
        max_position = max((pos.market_value / total_value for pos in positions), default=0)
        concentration_risk = max_position > profile["max_single_position"]
        
        # Calculate portfolio volatility (simplified)
        volatilities = [abs(pos.gain_loss_percent) / 100 for pos in positions]
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
        volatility_risk = avg_volatility > profile["volatility_limit"]
        
        return {
            "concentration_risk": concentration_risk,
            "max_position_weight": max_position,
            "volatility_risk": volatility_risk,
            "portfolio_volatility": avg_volatility,
            "risk_score": (int(concentration_risk) + int(volatility_risk)) / 2
        }
    
    def _generate_portfolio_recommendations(self, positions: List[PortfolioPosition], 
                                          risk_analysis: Dict[str, Any], 
                                          profile: Dict[str, Any]) -> List[str]:
        """Generate portfolio optimization recommendations."""
        recommendations = []
        
        if risk_analysis["concentration_risk"]:
            recommendations.append("Consider reducing position size in overweight holdings")
        
        if risk_analysis["volatility_risk"]:
            recommendations.append("Portfolio volatility exceeds risk tolerance - consider defensive positions")
        
        if len(positions) < 5:
            recommendations.append("Increase diversification by adding more positions")
        
        if not recommendations:
            recommendations.append("Portfolio is well-balanced for your risk profile")
        
        return recommendations


# =============================================================================
# 3. Healthcare Assistant Agent
# =============================================================================

@dataclass
class SymptomAssessment:
    """Represents a symptom assessment result."""
    symptoms: List[str]
    severity_score: int  # 1-10 scale
    urgency_level: str   # low, medium, high, emergency
    recommendations: List[str]
    specialist_referral: Optional[str]
    follow_up_needed: bool


class HealthcareAssistant:
    """
    Healthcare assistant agent for symptom checking, appointment scheduling,
    and health information. 
    
    DISCLAIMER: For educational purposes only. Not a substitute for professional medical advice.
    """
    
    def __init__(self):
        self.symptom_database = self._load_symptom_database()
        self.emergency_symptoms = self._load_emergency_symptoms()
        self.specialist_mapping = self._load_specialist_mapping()
    
    def _load_symptom_database(self) -> Dict[str, Dict[str, Any]]:
        """Load symptom assessment database."""
        return {
            "fever": {
                "severity_indicators": ["high_temperature", "chills", "sweating"],
                "common_causes": ["infection", "flu", "cold"],
                "urgency_factors": ["temperature_over_103", "persistent_fever"]
            },
            "headache": {
                "severity_indicators": ["severe_pain", "sudden_onset", "vision_changes"],
                "common_causes": ["tension", "migraine", "dehydration"],
                "urgency_factors": ["worst_headache_ever", "neurological_symptoms"]
            },
            "chest_pain": {
                "severity_indicators": ["severe_pain", "radiating_pain", "shortness_of_breath"],
                "common_causes": ["muscle_strain", "heartburn", "anxiety"],
                "urgency_factors": ["crushing_pain", "arm_pain", "sweating"]
            },
            "abdominal_pain": {
                "severity_indicators": ["severe_pain", "cramping", "bloating"],
                "common_causes": ["gas", "indigestion", "stress"],
                "urgency_factors": ["severe_cramping", "vomiting", "fever"]
            }
        }
    
    def _load_emergency_symptoms(self) -> List[str]:
        """Load list of emergency symptoms requiring immediate attention."""
        return [
            "chest_pain_with_shortness_of_breath",
            "severe_allergic_reaction",
            "difficulty_breathing",
            "loss_of_consciousness",
            "severe_bleeding",
            "signs_of_stroke",
            "severe_burns",
            "suspected_heart_attack"
        ]
    
    def _load_specialist_mapping(self) -> Dict[str, str]:
        """Load mapping of symptoms to medical specialists."""
        return {
            "heart_related": "cardiologist",
            "skin_related": "dermatologist", 
            "bone_joint_related": "orthopedist",
            "mental_health": "psychiatrist",
            "digestive_issues": "gastroenterologist",
            "respiratory_issues": "pulmonologist",
            "neurological_symptoms": "neurologist",
            "eye_related": "ophthalmologist"
        }
    
    @tool
    def symptom_checker(self, symptoms: List[str], severity: int = 5, 
                       duration: str = "recent") -> Dict[str, Any]:
        """
        Assess symptoms and provide health recommendations.
        
        DISCLAIMER: This is for informational purposes only and not medical advice.
        
        Args:
            symptoms: List of symptoms experienced
            severity: Pain/discomfort level on 1-10 scale
            duration: How long symptoms have persisted
            
        Returns:
            Dictionary with assessment and recommendations
        """
        # Check for emergency symptoms
        emergency_detected = any(
            symptom.lower() in " ".join(self.emergency_symptoms).lower() 
            for symptom in symptoms
        )
        
        if emergency_detected or severity >= 9:
            return {
                "urgency_level": "EMERGENCY",
                "recommendation": "SEEK IMMEDIATE MEDICAL ATTENTION",
                "call_911": True,
                "symptoms_assessed": symptoms,
                "severity": severity,
                "disclaimer": "This is not medical advice. Seek professional medical care."
            }
        
        # Assess individual symptoms
        assessment_results = []
        overall_urgency = "low"
        specialist_needed = None
        
        for symptom in symptoms:
            if symptom.lower() in self.symptom_database:
                symptom_data = self.symptom_database[symptom.lower()]
                
                # Determine urgency based on severity and symptom type
                if severity >= 7 or any(factor in symptom.lower() for factor in symptom_data.get("urgency_factors", [])):
                    symptom_urgency = "high"
                    overall_urgency = "high"
                elif severity >= 5:
                    symptom_urgency = "medium"
                    if overall_urgency == "low":
                        overall_urgency = "medium"
                else:
                    symptom_urgency = "low"
                
                assessment_results.append({
                    "symptom": symptom,
                    "urgency": symptom_urgency,
                    "common_causes": symptom_data.get("common_causes", []),
                    "severity_indicators": symptom_data.get("severity_indicators", [])
                })
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            assessment_results, severity, duration, overall_urgency
        )
        
        # Determine if specialist referral needed
        if overall_urgency == "high" or severity >= 7:
            specialist_needed = self._determine_specialist(symptoms)
        
        return {
            "urgency_level": overall_urgency,
            "severity_score": severity,
            "duration": duration,
            "symptoms_assessed": symptoms,
            "individual_assessments": assessment_results,
            "recommendations": recommendations,
            "specialist_referral": specialist_needed,
            "follow_up_timeframe": self._get_follow_up_timeframe(overall_urgency),
            "disclaimer": "This assessment is for informational purposes only. Consult healthcare professionals for medical advice.",
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    def _generate_health_recommendations(self, assessments: List[Dict], severity: int, 
                                       duration: str, urgency: str) -> List[str]:
        """Generate health recommendations based on assessment."""
        recommendations = []
        
        if urgency == "high":
            recommendations.extend([
                "Schedule appointment with healthcare provider within 24-48 hours",
                "Monitor symptoms closely for any worsening",
                "Keep a symptom diary with times and severity"
            ])
        elif urgency == "medium":
            recommendations.extend([
                "Consider scheduling appointment with healthcare provider within 1 week",
                "Rest and stay hydrated",
                "Monitor symptoms and note any changes"
            ])
        else:
            recommendations.extend([
                "Monitor symptoms for improvement over next few days",
                "Rest and maintain good self-care",
                "Contact healthcare provider if symptoms worsen or persist"
            ])
        
        # Duration-specific recommendations
        if "chronic" in duration.lower() or "weeks" in duration.lower():
            recommendations.append("Chronic symptoms warrant professional evaluation")
        
        return recommendations
    
    def _determine_specialist(self, symptoms: List[str]) -> Optional[str]:
        """Determine if specialist referral is needed."""
        symptoms_text = " ".join(symptoms).lower()
        
        for condition, specialist in self.specialist_mapping.items():
            condition_keywords = condition.replace("_", " ").split()
            if any(keyword in symptoms_text for keyword in condition_keywords):
                return specialist
        
        return "primary_care_physician"
    
    def _get_follow_up_timeframe(self, urgency: str) -> str:
        """Get recommended follow-up timeframe."""
        timeframes = {
            "emergency": "Immediate",
            "high": "24-48 hours",
            "medium": "1 week", 
            "low": "2 weeks or if symptoms worsen"
        }
        return timeframes.get(urgency, "As needed")


# =============================================================================
# 4. Research Automation Agent
# =============================================================================

class ResearchAutomationAgent:
    """
    Automated research agent for academic and business research,
    including literature review, data analysis, and report generation.
    """
    
    def __init__(self):
        self.research_databases = self._initialize_research_databases()
        self.citation_styles = self._load_citation_styles()
    
    def _initialize_research_databases(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock research databases."""
        return {
            "academic_papers": {
                "ai_machine_learning": [
                    {
                        "title": "Deep Learning Applications in Healthcare",
                        "authors": ["Smith, J.", "Johnson, A."],
                        "year": 2023,
                        "journal": "Journal of AI in Medicine",
                        "abstract": "This paper explores applications of deep learning in medical diagnosis...",
                        "citations": 45,
                        "doi": "10.1234/jaim.2023.001"
                    },
                    {
                        "title": "Transformer Networks for Natural Language Processing",
                        "authors": ["Brown, M.", "Davis, K."],
                        "year": 2022,
                        "journal": "Computational Linguistics Review",
                        "abstract": "Comprehensive review of transformer architectures...",
                        "citations": 128,
                        "doi": "10.1234/clr.2022.045"
                    }
                ],
                "financial_technology": [
                    {
                        "title": "Blockchain Applications in Modern Banking",
                        "authors": ["Wilson, R.", "Taylor, S."],
                        "year": 2023,
                        "journal": "FinTech Quarterly",
                        "abstract": "Analysis of blockchain implementation in banking systems...",
                        "citations": 23,
                        "doi": "10.1234/ftq.2023.012"
                    }
                ]
            },
            "market_reports": {
                "technology_trends": [
                    {
                        "title": "Global AI Market Analysis 2024",
                        "organization": "Tech Research Institute",
                        "year": 2024,
                        "summary": "The global AI market is expected to reach $1.8 trillion by 2030...",
                        "key_findings": ["70% growth in AI adoption", "Healthcare leading sector"]
                    }
                ]
            }
        }
    
    def _load_citation_styles(self) -> Dict[str, Dict[str, str]]:
        """Load different citation style formats."""
        return {
            "apa": {
                "format": "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}.",
                "example": "Smith, J. & Johnson, A. (2023). Deep Learning Applications. Journal of AI, 15(3), 45-67."
            },
            "mla": {
                "format": "{authors}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}.",
                "example": "Smith, John, and Alice Johnson. \"Deep Learning Applications.\" Journal of AI, vol. 15, no. 3, 2023, pp. 45-67."
            },
            "chicago": {
                "format": "{authors}. \"{title}.\" {journal} {volume}, no. {issue} ({year}): {pages}.",
                "example": "Smith, John, and Alice Johnson. \"Deep Learning Applications.\" Journal of AI 15, no. 3 (2023): 45-67."
            }
        }
    
    @tool
    def literature_search(self, query: str, field: str = "general", 
                         max_results: int = 10, year_range: str = "2020-2024") -> Dict[str, Any]:
        """
        Search academic literature and research papers.
        
        Args:
            query: Search terms and keywords
            field: Research field (ai_machine_learning, financial_technology, etc.)
            max_results: Maximum number of results to return
            year_range: Year range for search (e.g., "2020-2024")
            
        Returns:
            Dictionary with search results and metadata
        """
        query_terms = query.lower().split()
        results = []
        
        # Search in relevant databases
        if field in self.research_databases["academic_papers"]:
            papers = self.research_databases["academic_papers"][field]
            
            for paper in papers:
                relevance_score = 0
                searchable_text = (paper["title"] + " " + paper["abstract"]).lower()
                
                # Calculate relevance based on query terms
                for term in query_terms:
                    if term in searchable_text:
                        relevance_score += searchable_text.count(term)
                
                if relevance_score > 0:
                    results.append({
                        **paper,
                        "relevance_score": relevance_score,
                        "database": "academic_papers"
                    })
        
        # Sort by relevance and citation count
        results.sort(key=lambda x: (x["relevance_score"], x["citations"]), reverse=True)
        
        return {
            "query": query,
            "field": field,
            "results_found": len(results),
            "results": results[:max_results],
            "search_metadata": {
                "year_range": year_range,
                "databases_searched": ["academic_papers"],
                "search_timestamp": datetime.now().isoformat()
            }
        }
    
    @tool
    def generate_research_summary(self, papers: List[Dict[str, Any]], 
                                 focus_area: str = "general") -> Dict[str, Any]:
        """
        Generate research summary from collected papers.
        
        Args:
            papers: List of research papers to summarize
            focus_area: Specific area to focus the summary on
            
        Returns:
            Dictionary with research summary and insights
        """
        if not papers:
            return {"error": "No papers provided for summary"}
        
        # Analyze papers
        total_papers = len(papers)
        total_citations = sum(paper.get("citations", 0) for paper in papers)
        year_distribution = {}
        author_frequency = {}
        
        # Extract insights
        for paper in papers:
            year = paper.get("year", "Unknown")
            year_distribution[year] = year_distribution.get(year, 0) + 1
            
            authors = paper.get("authors", [])
            for author in authors:
                author_frequency[author] = author_frequency.get(author, 0) + 1
        
        # Generate key themes (simplified)
        key_themes = self._extract_key_themes(papers)
        
        # Generate summary
        summary = {
            "focus_area": focus_area,
            "papers_analyzed": total_papers,
            "total_citations": total_citations,
            "average_citations": total_citations / total_papers if total_papers > 0 else 0,
            "year_distribution": year_distribution,
            "key_themes": key_themes,
            "top_authors": sorted(author_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            "research_trends": self._identify_trends(papers, year_distribution),
            "recommendations": self._generate_research_recommendations(papers, key_themes),
            "summary_generated": datetime.now().isoformat()
        }
        
        return summary
    
    def _extract_key_themes(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from paper titles and abstracts."""
        all_text = ""
        for paper in papers:
            all_text += paper.get("title", "") + " " + paper.get("abstract", "") + " "
        
        # Simple keyword extraction (in production, use NLP libraries)
        common_terms = ["learning", "analysis", "system", "model", "data", "research", "study"]
        words = all_text.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 5 and word not in common_terms:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top themes
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _identify_trends(self, papers: List[Dict[str, Any]], year_dist: Dict) -> List[str]:
        """Identify research trends from the literature."""
        trends = []
        
        if len(year_dist) > 1:
            recent_years = [year for year in year_dist.keys() if isinstance(year, int) and year >= 2022]
            if recent_years:
                trends.append("Increasing research activity in recent years")
        
        avg_citations = sum(paper.get("citations", 0) for paper in papers) / len(papers)
        if avg_citations > 50:
            trends.append("High-impact research area with significant citations")
        
        if not trends:
            trends.append("Emerging research area with growing interest")
        
        return trends
    
    def _generate_research_recommendations(self, papers: List[Dict[str, Any]], 
                                         themes: List[Tuple[str, int]]) -> List[str]:
        """Generate recommendations for further research."""
        recommendations = []
        
        if len(papers) < 5:
            recommendations.append("Expand literature search to include more recent publications")
        
        if themes:
            top_theme = themes[0][0]
            recommendations.append(f"Focus deeper investigation on '{top_theme}' theme")
        
        recommendations.extend([
            "Consider cross-disciplinary perspectives",
            "Look for gaps in current research for potential contributions",
            "Analyze methodologies used across studies for best practices"
        ])
        
        return recommendations


# =============================================================================
# 5. Demonstration Functions
# =============================================================================

def demonstrate_customer_support():
    """Demonstrate customer support agent capabilities."""
    print("🎧 Customer Support Agent Demonstration")
    print("=" * 50)
    
    agent = CustomerSupportAgent()
    
    # Test FAQ search
    print("\n🔍 FAQ Search Test:")
    faq_result = agent.search_faq.invoke({
        "query": "how to reset my password",
        "category": "account"
    })
    
    print(f"Query: '{faq_result['query']}'")
    print(f"Results found: {faq_result['results_found']}")
    for result in faq_result['results'][:2]:
        print(f"  • {result['question']}: {result['answer'][:50]}...")
    
    # Test ticket creation
    print("\n🎫 Support Ticket Creation:")
    ticket_result = agent.create_support_ticket.invoke({
        "customer_id": "CUST001",
        "subject": "Urgent billing issue with my account",
        "description": "I was charged twice for my subscription this month",
        "category": "billing"
    })
    
    print(f"Ticket ID: {ticket_result['ticket_id']}")
    print(f"Priority: {ticket_result['priority']}")
    print(f"Assigned to: {ticket_result['assigned_agent']}")
    print(f"Response time: {ticket_result['estimated_response_time']}")


def demonstrate_financial_analysis():
    """Demonstrate financial analysis agent capabilities."""
    print("\n💰 Financial Analysis Agent Demonstration")
    print("=" * 50)
    
    agent = FinancialAnalysisAgent()
    
    # Test stock analysis
    print("\n📈 Stock Analysis:")
    stock_analysis = agent.get_stock_analysis.invoke({"symbol": "AAPL"})
    
    print(f"Stock: {stock_analysis['symbol']}")
    print(f"Price: ${stock_analysis['current_price']:.2f}")
    print(f"Change: {stock_analysis['daily_change_percent']:.2f}%")
    print(f"Recommendation: {stock_analysis['recommendation']['action']}")
    print(f"Confidence: {stock_analysis['recommendation']['confidence']:.1%}")
    
    # Test portfolio analysis
    print("\n📊 Portfolio Analysis:")
    sample_portfolio = [
        {"symbol": "AAPL", "quantity": 100, "avg_cost": 150.0},
        {"symbol": "GOOGL", "quantity": 50, "avg_cost": 2000.0},
        {"symbol": "MSFT", "quantity": 75, "avg_cost": 250.0}
    ]
    
    portfolio_analysis = agent.analyze_portfolio.invoke({
        "positions": sample_portfolio,
        "risk_profile": "moderate"
    })
    
    print(f"Portfolio value: ${portfolio_analysis['portfolio_value']:,.2f}")
    print(f"Total return: {portfolio_analysis['portfolio_metrics']['total_return_percent']:.2f}%")
    print(f"Risk score: {portfolio_analysis['risk_analysis']['risk_score']:.1f}")
    print(f"Recommendations: {len(portfolio_analysis['recommendations'])}")


def demonstrate_healthcare_assistant():
    """Demonstrate healthcare assistant capabilities."""
    print("\n🏥 Healthcare Assistant Demonstration")
    print("=" * 50)
    
    agent = HealthcareAssistant()
    
    # Test symptom checker
    print("\n🩺 Symptom Assessment:")
    assessment = agent.symptom_checker.invoke({
        "symptoms": ["headache", "fever"],
        "severity": 6,
        "duration": "2 days"
    })
    
    print(f"Symptoms: {', '.join(assessment['symptoms_assessed'])}")
    print(f"Urgency level: {assessment['urgency_level']}")
    print(f"Severity score: {assessment['severity_score']}/10")
    print(f"Follow-up: {assessment['follow_up_timeframe']}")
    print(f"Recommendations: {len(assessment['recommendations'])}")
    
    for rec in assessment['recommendations'][:2]:
        print(f"  • {rec}")


def demonstrate_research_automation():
    """Demonstrate research automation capabilities."""
    print("\n🔬 Research Automation Agent Demonstration")
    print("=" * 50)
    
    agent = ResearchAutomationAgent()
    
    # Test literature search
    print("\n📚 Literature Search:")
    search_results = agent.literature_search.invoke({
        "query": "deep learning applications",
        "field": "ai_machine_learning",
        "max_results": 3
    })
    
    print(f"Query: '{search_results['query']}'")
    print(f"Results found: {search_results['results_found']}")
    
    for result in search_results['results']:
        print(f"  • {result['title']} ({result['year']})")
        print(f"    Citations: {result['citations']}, Relevance: {result['relevance_score']}")
    
    # Test research summary
    print("\n📝 Research Summary:")
    if search_results['results']:
        summary = agent.generate_research_summary.invoke({
            "papers": search_results['results'],
            "focus_area": "AI applications"
        })
        
        print(f"Papers analyzed: {summary['papers_analyzed']}")
        print(f"Average citations: {summary['average_citations']:.1f}")
        print(f"Key themes: {len(summary['key_themes'])}")
        print(f"Research trends: {len(summary['research_trends'])}")


# Main demonstration
if __name__ == "__main__":
    print("🎓 Module 6: Real-World Applications")
    print("=" * 60)
    
    # Demonstrate all real-world applications
    demonstrate_customer_support()
    demonstrate_financial_analysis()
    demonstrate_healthcare_assistant()
    demonstrate_research_automation()
    
    print("\n✅ Module 6 demonstrations completed!")
    print("🎉 All course modules are now available!")
    print("\n🚀 Ready for production deployment and real-world use!")
