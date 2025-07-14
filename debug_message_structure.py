#!/usr/bin/env python3
"""
调试消息结构的专用脚本
在这里设置断点来检查实际环境中的消息结构
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from set_api_keys import set_api_keys, set_http_proxy

# 设置API密钥
set_api_keys()
set_http_proxy()

def debug_message_structure():
    """专门用于调试消息结构的函数"""
    print("=== 开始调试消息结构 ===")
    
    # 创建简化的配置
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1  # 减少轮次以便快速调试
    config["max_risk_discuss_rounds"] = 1
    
    # 只使用基本面分析师，减少复杂性
    analysts = [AnalystType.FUNDAMENTALS.value]
    
    # 初始化图
    graph = TradingAgentsGraph(analysts, config=config, debug=True)
    
    # 创建初始状态
    init_state = graph.propagator.create_initial_state("BTBT", "2025-07-10")
    
    # 获取图参数
    args = graph.propagator.get_graph_args()
    args["stream_mode"] = "updates"
    
    print("开始流式处理...")
    
    # 用于追踪已处理的消息
    last_processed_message = None
    chunk_count = 0
    
    # 流式处理
    for chunk in graph.graph.stream(init_state, **args):
        chunk_count += 1
        print(f"\n--- Chunk {chunk_count} ---")
        
        # 获取agent名称
        if not chunk or "__end__" in chunk:
            continue
            
        agent_name = list(chunk.keys())[0]
        print(f"Agent: {agent_name}")
        
        # 获取消息
        if not isinstance(chunk[agent_name], dict):
            continue
            
        messages_in_chunk = chunk[agent_name].get("messages", [])
        print(f"Messages count: {len(messages_in_chunk)}")
        
        # 🔴 在这里设置断点！检查 messages_in_chunk 的结构
        if len(messages_in_chunk) > 0:
            print("=== 消息结构调试点 ===")
            
            for i, msg in enumerate(messages_in_chunk):
                print(f"\nMessage {i}:")
                print(f"  Type: {type(msg)}")
                print(f"  Has ID: {hasattr(msg, 'id')}")
                if hasattr(msg, 'id'):
                    print(f"  ID: {msg.id}")
                print(f"  Has content: {hasattr(msg, 'content')}")
                if hasattr(msg, 'content'):
                    content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                    print(f"  Content preview: {content_preview}")
                print(f"  Has tool_calls: {hasattr(msg, 'tool_calls')}")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  Tool calls count: {len(msg.tool_calls)}")
                    for j, tool_call in enumerate(msg.tool_calls):
                        print(f"    Tool {j}: {tool_call}")
                
                # 🔴 在这里设置断点！检查具体消息对象
                # 在调试器中可以展开 msg 对象查看所有属性
                debug_point = msg  # 设置断点在这行
                
            # 🔴 在这里设置断点！检查消息处理逻辑
            if last_processed_message is not None:
                print(f"\nLast processed message type: {type(last_processed_message)}")
                if hasattr(last_processed_message, 'id'):
                    print(f"Last processed message ID: {last_processed_message.id}")
            
            # 更新追踪信息
            last_processed_message = messages_in_chunk[-1]
            
        # 限制处理的chunk数量，避免调试时间过长
        if chunk_count >= 5:
            print("达到调试限制，停止处理")
            break
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_message_structure() 