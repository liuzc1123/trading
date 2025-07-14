from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align

from cli.message_buffer import MessageBuffer

console = Console()


class DisplayManager:
    """管理CLI界面显示的类"""
    
    def __init__(self, message_buffer: MessageBuffer):
        self.message_buffer = message_buffer
    
    def create_layout(self) -> Layout:
        """创建界面布局"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_column(
            Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
        )
        layout["upper"].split_row(
            Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
        )
        return layout
    
    def update_display(self, layout: Layout, spinner_text: Optional[str] = None):
        """更新整个显示界面"""
        self._update_header(layout)
        self._update_progress(layout)
        self._update_messages(layout, spinner_text)
        self._update_analysis(layout)
        self._update_footer(layout)
    
    def _update_header(self, layout: Layout):
        """更新头部面板"""
        layout["header"].update(
            Panel(
                "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
                "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
                title="Welcome to TradingAgents",
                border_style="green",
                padding=(1, 2),
                expand=True,
            )
        )
    
    def _update_progress(self, layout: Layout):
        """更新进度面板"""
        progress_table = Table(
            show_header=True,
            header_style="bold magenta",
            show_footer=False,
            box=box.SIMPLE_HEAD,
            title=None,
            padding=(0, 2),
            expand=True,
        )
        progress_table.add_column("Team", style="cyan", justify="center", width=20)
        progress_table.add_column("Agent", style="green", justify="center", width=20)
        progress_table.add_column("Status", style="yellow", justify="center", width=20)
        
        # 团队配置
        teams = {
            "Analyst Team": [
                "Market Analyst",
                "Social Analyst", 
                "News Analyst",
                "Fundamentals Analyst",
            ],
            "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
            "Trading Team": ["Trader"],
            "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
            "Portfolio Management": ["Portfolio Manager"],
        }
        
        for team, agents in teams.items():
            # 添加第一个代理和团队名称
            first_agent = agents[0]
            status_cell = self._get_status_cell(self.message_buffer.agent_status[first_agent])
            progress_table.add_row(team, first_agent, status_cell)
            
            # 添加团队中其余代理
            for agent in agents[1:]:
                status_cell = self._get_status_cell(self.message_buffer.agent_status[agent])
                progress_table.add_row("", agent, status_cell)
            
            # 在每个团队后添加分隔线
            progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")
        
        layout["progress"].update(
            Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
        )
    
    def _get_status_cell(self, status: str):
        """获取状态显示单元格"""
        if status == "in_progress":
            return Spinner("dots", text="[blue]in_progress[/blue]", style="bold cyan")
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green", 
                "error": "red",
            }.get(status, "white")
            return f"[{status_color}]{status}[/{status_color}]"
    
    def _update_messages(self, layout: Layout, spinner_text: Optional[str] = None):
        """更新消息面板"""
        messages_table = Table(
            show_header=True,
            header_style="bold magenta",
            show_footer=False,
            expand=True,
            box=box.MINIMAL,
            show_lines=True,
            padding=(0, 1),
        )
        messages_table.add_column("Time", style="cyan", width=8, justify="center")
        messages_table.add_column("Type", style="green", width=10, justify="center")
        messages_table.add_column("Content", style="white", no_wrap=False, ratio=1)
        
        # 合并工具调用和消息
        all_messages = self._combine_messages()
        
        # 计算可显示的消息数量
        max_messages = 12
        recent_messages = all_messages[-max_messages:]
        
        # 添加消息到表格
        for timestamp, msg_type, content in recent_messages:
            wrapped_content = Text(content, overflow="fold")
            messages_table.add_row(timestamp, msg_type, wrapped_content)
        
        if spinner_text:
            messages_table.add_row("", "Spinner", spinner_text)
        
        # 添加截断提示
        if len(all_messages) > max_messages:
            messages_table.footer = (
                f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
            )
        
        layout["messages"].update(
            Panel(
                messages_table,
                title="Messages & Tools",
                border_style="blue",
                padding=(1, 2),
            )
        )
    
    def _combine_messages(self) -> list:
        """合并工具调用和普通消息"""
        all_messages = []
        
        # 添加工具调用
        for timestamp, tool_name, args in self.message_buffer.tool_calls:
            if isinstance(args, str) and len(args) > 100:
                args = args[:97] + "..."
            all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))
        
        # 添加普通消息
        for timestamp, msg_type, content in self.message_buffer.messages:
            content_str = self._extract_content_string(content)
            if len(content_str) > 200:
                content_str = content_str[:197] + "..."
            all_messages.append((timestamp, msg_type, content_str))
        
        # 按时间戳排序
        all_messages.sort(key=lambda x: x[0])
        return all_messages
    
    def _extract_content_string(self, content) -> str:
        """提取字符串内容"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            return ' '.join(text_parts)
        else:
            return str(content)
    
    def _update_analysis(self, layout: Layout):
        """更新分析面板"""
        if self.message_buffer.current_report:
            layout["analysis"].update(
                Panel(
                    Markdown(self.message_buffer.current_report),
                    title="Current Report",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            layout["analysis"].update(
                Panel(
                    "[italic]Waiting for analysis report...[/italic]",
                    title="Current Report",
                    border_style="green",
                    padding=(1, 2),
                )
            )
    
    def _update_footer(self, layout: Layout):
        """更新页脚"""
        tool_calls_count = len(self.message_buffer.tool_calls)
        llm_calls_count = sum(
            1 for _, msg_type, _ in self.message_buffer.messages 
            if msg_type.startswith("LLM")
        )
        reports_count = sum(
            1 for content in self.message_buffer.report_sections.values() 
            if content is not None
        )
        
        stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        stats_table.add_column("Stats", justify="center")
        stats_table.add_row(
            f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
        )
        
        layout["footer"].update(Panel(stats_table, border_style="grey50"))
    
    def display_complete_report(self, final_state: dict):
        """显示完整的分析报告"""
        console.print("\n[bold green]Complete Analysis Report[/bold green]\n")
        
        # I. 分析师团队报告
        self._display_analyst_reports(final_state)
        
        # II. 研究团队报告
        self._display_research_reports(final_state)
        
        # III. 交易团队报告
        self._display_trading_reports(final_state)
        
        # IV. 风险管理团队报告
        self._display_risk_reports(final_state)
    
    def _display_analyst_reports(self, final_state: dict):
        """显示分析师团队报告"""
        analyst_reports = []
        
        report_configs = [
            ("market_report", "Market Analyst"),
            ("sentiment_report", "Social Analyst"),
            ("news_report", "News Analyst"),
            ("fundamentals_report", "Fundamentals Analyst"),
        ]
        
        for key, title in report_configs:
            if final_state.get(key):
                analyst_reports.append(
                    Panel(
                        Markdown(final_state[key]),
                        title=title,
                        border_style="blue",
                        padding=(1, 2),
                    )
                )
        
        if analyst_reports:
            console.print(
                Panel(
                    Columns(analyst_reports, equal=True, expand=True),
                    title="I. Analyst Team Reports",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
    
    def _display_research_reports(self, final_state: dict):
        """显示研究团队报告"""
        if final_state.get("investment_debate_state"):
            research_reports = []
            debate_state = final_state["investment_debate_state"]
            
            report_configs = [
                ("bull_history", "Bull Researcher"),
                ("bear_history", "Bear Researcher"),
                ("judge_decision", "Research Manager"),
            ]
            
            for key, title in report_configs:
                if debate_state.get(key):
                    research_reports.append(
                        Panel(
                            Markdown(debate_state[key]),
                            title=title,
                            border_style="blue",
                            padding=(1, 2),
                        )
                    )
            
            if research_reports:
                console.print(
                    Panel(
                        Columns(research_reports, equal=True, expand=True),
                        title="II. Research Team Decision",
                        border_style="magenta",
                        padding=(1, 2),
                    )
                )
    
    def _display_trading_reports(self, final_state: dict):
        """显示交易团队报告"""
        if final_state.get("trader_investment_plan"):
            console.print(
                Panel(
                    Panel(
                        Markdown(final_state["trader_investment_plan"]),
                        title="Trader",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="III. Trading Team Plan",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
    
    def _display_risk_reports(self, final_state: dict):
        """显示风险管理团队报告"""
        if final_state.get("risk_debate_state"):
            risk_reports = []
            risk_state = final_state["risk_debate_state"]
            
            report_configs = [
                ("risky_history", "Aggressive Analyst"),
                ("safe_history", "Conservative Analyst"),
                ("neutral_history", "Neutral Analyst"),
            ]
            
            for key, title in report_configs:
                if risk_state.get(key):
                    risk_reports.append(
                        Panel(
                            Markdown(risk_state[key]),
                            title=title,
                            border_style="blue",
                            padding=(1, 2),
                        )
                    )
            
            if risk_reports:
                console.print(
                    Panel(
                        Columns(risk_reports, equal=True, expand=True),
                        title="IV. Risk Management Team Decision",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
            
            # V. 投资组合经理决策
            if risk_state.get("judge_decision"):
                console.print(
                    Panel(
                        Panel(
                            Markdown(risk_state["judge_decision"]),
                            title="Portfolio Manager",
                            border_style="blue",
                            padding=(1, 2),
                        ),
                        title="V. Portfolio Manager Decision",
                        border_style="green",
                        padding=(1, 2),
                    )
                ) 