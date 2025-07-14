import typer
from rich.console import Console

from set_api_keys import set_api_keys, set_http_proxy
from cli.message_buffer import MessageBuffer
from cli.display import DisplayManager
from cli.user_interface import UserInterface
from cli.analysis_runner import AnalysisRunner

# 设置API密钥和代理
set_api_keys()
set_http_proxy()

console = Console()

# 创建Typer应用
app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,
)


def run_analysis():
    """运行分析的主函数"""
    # 创建核心组件
    message_buffer = MessageBuffer()
    display_manager = DisplayManager(message_buffer)
    user_interface = UserInterface()
    analysis_runner = AnalysisRunner(message_buffer, display_manager)
    
    try:
        # 获取用户选择
        selections = user_interface.get_user_selections()
        
        # 运行分析
        analysis_runner.run_analysis(selections)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during analysis: {str(e)}[/red]")
        raise


@app.command()
def analyze():
    """分析命令"""
    run_analysis()


if __name__ == "__main__":
    app()
