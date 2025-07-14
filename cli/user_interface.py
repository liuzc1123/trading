import datetime
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

from cli.models import AnalystType
from cli.utils import *

console = Console()


class UserInterface:
    """用户界面类，处理所有用户交互"""
    
    def __init__(self):
        self.console = console
    
    def get_user_selections(self) -> dict:
        """获取用户的所有选择"""
        self._display_welcome()
        
        # 按正确顺序获取选择
        ticker = self._get_ticker()
        analysis_date = self._get_analysis_date()
        analysts = self._select_analysts()
        research_depth = self._select_research_depth()
        llm_provider, backend_url = self._select_llm_provider()
        shallow_thinker, deep_thinker = self._get_thinking_agents(llm_provider)
        
        return {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "analysts": analysts,
            "research_depth": research_depth,
            "llm_provider": llm_provider.lower(),
            "backend_url": backend_url,
            "shallow_thinker": shallow_thinker,
            "deep_thinker": deep_thinker,
        }
    
    def _display_welcome(self):
        """显示欢迎界面"""
        # 显示ASCII艺术欢迎消息
        with open("./cli/static/welcome.txt", "r") as f:
            welcome_ascii = f.read()
        
        # 创建欢迎框内容
        welcome_content = f"{welcome_ascii}\n"
        welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
        welcome_content += "[bold]Workflow Steps:[/bold]\n"
        welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
        welcome_content += "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
        
        # 创建并居中欢迎框
        welcome_box = Panel(
            welcome_content,
            border_style="green",
            padding=(1, 2),
            title="Welcome to TradingAgents",
            subtitle="Multi-Agents LLM Financial Trading Framework",
        )
        self.console.print(Align.center(welcome_box))
        self.console.print()  # 添加空行
    
    def _create_question_box(self, title: str, prompt: str, default: str = None) -> Panel:
        """创建问题框"""
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))
    
    def _get_ticker(self) -> str:
        """获取股票代码"""
        self.console.print(
            self._create_question_box(
                "Step 1: Ticker Symbol", 
                "Enter the ticker symbol to analyze", 
                "SPY"
            )
        )
        return typer.prompt("", default="SPY")
    
    def _get_analysis_date(self) -> str:
        """获取分析日期"""
        default_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.console.print(
            self._create_question_box(
                "Step 2: Analysis Date",
                "Enter the analysis date (YYYY-MM-DD)",
                default_date,
            )
        )
        
        while True:
            date_str = typer.prompt("", default=default_date)
            try:
                # 验证日期格式并确保不是未来日期
                analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                if analysis_date.date() > datetime.datetime.now().date():
                    self.console.print("[red]Error: Analysis date cannot be in the future[/red]")
                    continue
                return date_str
            except ValueError:
                self.console.print("[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]")
    
    def _select_analysts(self) -> list:
        """选择分析师"""
        self.console.print(
            self._create_question_box(
                "Step 3: Analysts Team", 
                "Select your LLM analyst agents for the analysis"
            )
        )
        selected_analysts = select_analysts()
        self.console.print(
            f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
        )
        return selected_analysts
    
    def _select_research_depth(self) -> int:
        """选择研究深度"""
        self.console.print(
            self._create_question_box(
                "Step 4: Research Depth", 
                "Select your research depth level"
            )
        )
        return select_research_depth()
    
    def _select_llm_provider(self) -> tuple:
        """选择LLM提供商"""
        self.console.print(
            self._create_question_box(
                "Step 5: OpenAI backend", 
                "Select which service to talk to"
            )
        )
        return select_llm_provider()
    
    def _get_thinking_agents(self, llm_provider: str) -> tuple:
        """获取思考代理"""
        self.console.print(
            self._create_question_box(
                "Step 6: Thinking Agents", 
                "Select your thinking agents for analysis"
            )
        )
        shallow_thinker = select_shallow_thinking_agent(llm_provider)
        deep_thinker = select_deep_thinking_agent(llm_provider)
        return shallow_thinker, deep_thinker 