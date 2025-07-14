import datetime
from collections import deque
from typing import Optional, Dict, Any


class MessageBuffer:
    """消息缓冲区类，用于管理消息、工具调用和报告状态"""
    
    def __init__(self, max_length: int = 100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report: Optional[str] = None
        self.final_report: Optional[str] = None
        
        # 代理状态
        self.agent_status = {
            # 分析师团队
            "Market Analyst": "pending",
            "Social Analyst": "pending", 
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            # 研究团队
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # 交易团队
            "Trader": "pending",
            # 风险管理团队
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # 投资组合管理团队
            "Portfolio Manager": "pending",
        }
        
        self.current_agent: Optional[str] = None
        
        # 报告部分
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }
    
    def add_message(self, message_type: str, content: Any):
        """添加消息到缓冲区"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))
    
    def add_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """添加工具调用到缓冲区"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))
    
    def update_agent_status(self, agent: str, status: str):
        """更新代理状态"""
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent
    
    def update_report_section(self, section_name: str, content: str):
        """更新报告部分"""
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()
    
    def _update_current_report(self):
        """更新当前报告显示"""
        # 找到最近更新的部分
        latest_section = None
        latest_content = None
        
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
        
        if latest_section and latest_content:
            # 格式化当前部分用于显示
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )
        
        # 更新完整的最终报告
        self._update_final_report()
    
    def _update_final_report(self):
        """更新完整的最终报告"""
        report_parts = []
        
        # 分析师团队报告
        analyst_sections = [
            "market_report", 
            "sentiment_report", 
            "news_report", 
            "fundamentals_report"
        ]
        
        if any(self.report_sections[section] for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            
            section_mappings = [
                ("market_report", "Market Analysis"),
                ("sentiment_report", "Social Sentiment"),
                ("news_report", "News Analysis"),
                ("fundamentals_report", "Fundamentals Analysis"),
            ]
            
            for section_key, section_title in section_mappings:
                if self.report_sections[section_key]:
                    report_parts.append(
                        f"### {section_title}\n{self.report_sections[section_key]}"
                    )
        
        # 研究团队报告
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")
        
        # 交易团队报告
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")
        
        # 投资组合管理决策
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")
        
        self.final_report = "\n\n".join(report_parts) if report_parts else None 