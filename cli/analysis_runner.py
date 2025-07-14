from typing import Dict, Any
from pathlib import Path
from functools import wraps
from rich.live import Live
from langchain_core.messages import ToolMessage

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from .message_buffer import MessageBuffer
from .display import DisplayManager
from .utils import extract_content_string


class AnalysisRunner:
    """分析运行器类，负责执行分析流程"""
    
    def __init__(self, message_buffer: MessageBuffer, display_manager: DisplayManager):
        self.message_buffer = message_buffer
        self.display_manager = display_manager
        self.graph = None
        
    def run_analysis(self, selections: Dict[str, Any]):
        """运行分析流程"""
        # 创建配置
        config = self._create_config(selections)
        
        # 初始化图
        self.graph = TradingAgentsGraph(
            [analyst.value for analyst in selections["analysts"]], 
            config=config, 
            debug=True
        )
        
        # 创建结果目录
        results_dir, log_file = self._setup_directories(selections, config)
        
        # 设置装饰器
        self._setup_decorators(results_dir, log_file)
        
        # 创建布局并开始显示
        layout = self.display_manager.create_layout()
        
        with Live(layout, refresh_per_second=4) as live:
            self._initialize_display(layout, selections)
            self._execute_analysis(layout, selections)
    
    def _create_config(self, selections: Dict[str, Any]) -> Dict[str, Any]:
        """创建配置"""
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = selections["research_depth"]
        config["max_risk_discuss_rounds"] = selections["research_depth"]
        config["quick_think_llm"] = selections["shallow_thinker"]
        config["deep_think_llm"] = selections["deep_thinker"]
        config["backend_url"] = selections["backend_url"]
        config["llm_provider"] = selections["llm_provider"]
        return config
    
    def _setup_directories(self, selections: Dict[str, Any], config: Dict[str, Any]) -> tuple:
        """设置目录和日志文件"""
        results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
        results_dir.mkdir(parents=True, exist_ok=True)
        
        report_dir = results_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = results_dir / "message_tool.log"
        log_file.touch(exist_ok=True)
        
        return results_dir, log_file
    
    def _setup_decorators(self, results_dir: Path, log_file: Path):
        """设置装饰器用于保存消息和报告"""
        report_dir = results_dir / "reports"
        
        # 保存消息装饰器
        def save_message_decorator(obj, func_name):
            func = getattr(obj, func_name)
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                timestamp, message_type, content = obj.messages[-1]
                content = content.replace("\n", " ")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} [{message_type}] {content}\n")
                return result
            return wrapper
        
        # 保存工具调用装饰器
        def save_tool_call_decorator(obj, func_name):
            func = getattr(obj, func_name)
            @wraps(func)
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)
                timestamp, tool_name, args = obj.tool_calls[-1]
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
            return wrapper
        
        # 保存报告部分装饰器
        def save_report_section_decorator(obj, func_name):
            func = getattr(obj, func_name)
            @wraps(func)
            def wrapper(section_name, content):
                func(section_name, content)
                if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                    content = obj.report_sections[section_name]
                    if content:
                        file_name = f"{section_name}.md"
                        with open(report_dir / file_name, "w", encoding="utf-8") as f:
                            f.write(content)
            return wrapper
        
        # 应用装饰器
        self.message_buffer.add_message = save_message_decorator(self.message_buffer, "add_message")
        self.message_buffer.add_tool_call = save_tool_call_decorator(self.message_buffer, "add_tool_call")
        self.message_buffer.update_report_section = save_report_section_decorator(self.message_buffer, "update_report_section")
    
    def _initialize_display(self, layout, selections: Dict[str, Any]):
        """初始化显示"""
        self.display_manager.update_display(layout)
        
        # 添加初始消息
        self.message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        self.message_buffer.add_message("System", f"Analysis date: {selections['analysis_date']}")
        self.message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}"
        )
        
        # 重置状态
        self._reset_states()
        
        # 设置第一个分析师为进行中
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        self.message_buffer.update_agent_status(first_analyst, "in_progress")
        
        self.display_manager.update_display(layout)
    
    def _reset_states(self):
        """重置所有状态"""
        # 重置代理状态
        for agent in self.message_buffer.agent_status:
            self.message_buffer.update_agent_status(agent, "pending")
        
        # 重置报告部分
        for section in self.message_buffer.report_sections:
            self.message_buffer.report_sections[section] = None
        
        self.message_buffer.current_report = None
        self.message_buffer.final_report = None
    
    def _execute_analysis(self, layout, selections: Dict[str, Any]):
        """执行分析"""
        # 创建spinner文本
        spinner_text = f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        self.display_manager.update_display(layout, spinner_text)
        
        # 初始化状态和获取图参数
        init_agent_state = self.graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = self.graph.propagator.get_graph_args()
        args["stream_mode"] = "updates"
        
        # 流式分析
        trace = []
        last_processed_message = None
        
        for chunk in self.graph.graph.stream(init_agent_state, **args):
            agent_name = self._get_agent_name_from_chunk(chunk)
            
            if agent_name == "__end__" or not isinstance(chunk[agent_name], dict):
                continue
            
            node_output = chunk[agent_name]
            messages_in_chunk = node_output.get("messages", [])
            
            if len(messages_in_chunk) > 0:
                # 处理新消息
                messages_to_process = self._get_new_messages(messages_in_chunk, last_processed_message)
                
                self._process_messages(messages_to_process, agent_name, selections)
                
                if messages_to_process:
                    last_processed_message = messages_to_process[-1]
                
                # 处理报告更新
                self._process_report_updates(node_output, selections)
                
                # 更新显示
                self.display_manager.update_display(layout)
            
            trace.append(chunk)
        
        # 完成分析
        self._finalize_analysis(layout, selections, trace)
    
    def _get_agent_name_from_chunk(self, chunk: dict) -> str:
        """从块中获取代理名称"""
        if not chunk:
            return "System"
        return list(chunk.keys())[0]
    
    def _get_new_messages(self, messages_in_chunk: list, last_processed_message) -> list:
        """获取需要处理的新消息"""
        if last_processed_message is None:
            return messages_in_chunk
        
        start_index = 0
        for i, msg in enumerate(messages_in_chunk):
            # 尝试通过ID匹配
            if hasattr(msg, 'id') and hasattr(last_processed_message, 'id'):
                if msg.id == last_processed_message.id:
                    start_index = i + 1
                    break
            # 如果没有ID，尝试通过内容和类型匹配
            elif (type(msg) == type(last_processed_message) and 
                  hasattr(msg, 'content') and hasattr(last_processed_message, 'content') and
                  msg.content == last_processed_message.content):
                start_index = i + 1
                break
        
        return messages_in_chunk[start_index:]
    
    def _process_messages(self, messages_to_process: list, agent_name: str, selections: Dict[str, Any]):
        """处理消息"""
        for message in messages_to_process:
            if isinstance(message, ToolMessage):
                content = extract_content_string(message.content)
                msg_type = "Tool Return"
            elif hasattr(message, "content"):
                content = extract_content_string(message.content)
                msg_type = f'LLM: {agent_name}'
            else:
                content = str(message)
                msg_type = "System"
            
            self.message_buffer.add_message(msg_type, content)
            
            # 记录工具调用
            if hasattr(message, "tool_calls"):
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        self.message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                    else:
                        self.message_buffer.add_tool_call(tool_call.name, tool_call.args)
    
    def _process_report_updates(self, node_output: dict, selections: Dict[str, Any]):
        """处理报告更新"""
        # 分析师团队报告
        analyst_reports = [
            ("market_report", "Market Analyst"),
            ("sentiment_report", "Social Analyst"),
            ("news_report", "News Analyst"),
            ("fundamentals_report", "Fundamentals Analyst"),
        ]
        
        for report_key, analyst_name in analyst_reports:
            if report_key in node_output and node_output[report_key]:
                self.message_buffer.update_report_section(report_key, node_output[report_key])
                self.message_buffer.update_agent_status(analyst_name, "completed")
                self._set_next_analyst_in_progress(report_key, selections)
        
        # 研究团队
        self._process_research_team_updates(node_output)
        
        # 交易团队
        if "trader_investment_plan" in node_output and node_output["trader_investment_plan"]:
            self.message_buffer.update_report_section("trader_investment_plan", f'### Trader plan\n{node_output["trader_investment_plan"]}')
            self.message_buffer.update_agent_status("Risky Analyst", "in_progress")
        
        # 风险管理团队
        self._process_risk_team_updates(node_output)
    
    def _set_next_analyst_in_progress(self, current_report_key: str, selections: Dict[str, Any]):
        """设置下一个分析师为进行中状态"""
        next_analyst_map = {
            "market_report": ("social", "Social Analyst"),
            "sentiment_report": ("news", "News Analyst"),
            "news_report": ("fundamentals", "Fundamentals Analyst"),
        }
        
        if current_report_key in next_analyst_map:
            analyst_type, analyst_name = next_analyst_map[current_report_key]
            if analyst_type in [a.value for a in selections["analysts"]]:
                self.message_buffer.update_agent_status(analyst_name, "in_progress")
            elif current_report_key == "fundamentals_report":
                self._update_research_team_status("in_progress")
    
    def _process_research_team_updates(self, node_output: dict):
        """处理研究团队更新"""
        if "investment_debate_state" in node_output and node_output["investment_debate_state"]:
            debate_state = node_output["investment_debate_state"]
            
            if "bull_history" in debate_state or "bear_history" in debate_state:
                self._update_research_team_status("in_progress")
            
            if "judge_decision" in debate_state and debate_state["judge_decision"]:
                investment_plan = (
                    f"### Bull/Bear Debate\n{debate_state['history']}\n"
                    f"### Portfolio Manager Decision: \n{debate_state['judge_decision']}"
                )
                self.message_buffer.update_report_section("investment_plan", investment_plan)
                self._update_research_team_status("completed")
                self.message_buffer.update_agent_status("Trader", "in_progress")
    
    def _process_risk_team_updates(self, node_output: dict):
        """处理风险管理团队更新"""
        if "risk_debate_state" in node_output and node_output["risk_debate_state"]:
            risk_state = node_output["risk_debate_state"]
            
            if any(key in risk_state and risk_state[key] for key in ["risky_history", "safe_history", "neutral_history"]):
                for analyst in ["Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
                    self.message_buffer.update_agent_status(analyst, "in_progress")
            
            if "judge_decision" in risk_state and risk_state["judge_decision"]:
                self.message_buffer.update_report_section(
                    "final_trade_decision", 
                    f"### Portfolio Manager Decision\n{risk_state['judge_decision']}"
                )
                for analyst in ["Risky Analyst", "Safe Analyst", "Neutral Analyst", "Portfolio Manager"]:
                    self.message_buffer.update_agent_status(analyst, "completed")
    
    def _update_research_team_status(self, status: str):
        """更新研究团队状态"""
        research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
        for agent in research_team:
            self.message_buffer.update_agent_status(agent, status)
    
    def _finalize_analysis(self, layout, selections: Dict[str, Any], trace: list):
        """完成分析"""
        # 标记所有代理为完成
        for agent in self.message_buffer.agent_status:
            self.message_buffer.update_agent_status(agent, "completed")
        
        self.message_buffer.add_message("Analysis", f"Completed analysis for {selections['analysis_date']}")
        
        # 重建最终状态
        final_state = {}
        for chunk in trace:
            for key, value in chunk.items():
                if key != "__end__":
                    final_state.update(value)
        
        # 处理最终决策
        decision = self.graph.process_signal(final_state["final_trade_decision"])
        
        # 显示完整报告
        self.display_manager.display_complete_report(final_state)
        self.display_manager.update_display(layout) 