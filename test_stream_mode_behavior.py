#!/usr/bin/env python3
"""
测试stream_mode="updates"的具体行为
验证messages列表的粒度：是包含所有历史消息还是只有新增消息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量避免导入问题
os.environ['SKIP_DEEPSEEK'] = '1'

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from set_api_keys import set_api_keys, set_http_proxy

# 设置API密钥
set_api_keys()
set_http_proxy()

def test_stream_mode_behavior():
    """测试stream_mode="updates"的具体行为"""
    print("=== 测试 stream_mode='updates' 行为 ===")
    
    # 创建简化的配置
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["llm_provider"] = "openai"  # 使用openai避免导入问题
    
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
    
    chunk_count = 0
    all_messages_seen = []  # 记录所有见过的消息
    
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
        print(f"Messages count in this chunk: {len(messages_in_chunk)}")
        
        # 🔴 关键测试：检查messages列表的内容
        if len(messages_in_chunk) > 0:
            print("=== Messages 内容分析 ===")
            
            # 检查是否包含历史消息
            current_messages = []
            for i, msg in enumerate(messages_in_chunk):
                if hasattr(msg, 'content'):
                    content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                    current_messages.append(content_preview)
                    print(f"  Message {i}: {content_preview}")
            
            # 检查是否有重复消息
            new_messages = []
            for msg in current_messages:
                if msg not in all_messages_seen:
                    new_messages.append(msg)
                    all_messages_seen.append(msg)
            
            print(f"  新增消息数量: {len(new_messages)}")
            print(f"  重复消息数量: {len(current_messages) - len(new_messages)}")
            
            if len(new_messages) > 0:
                print("  新增消息内容:")
                for msg in new_messages:
                    print(f"    - {msg}")
            
            # 检查chunk中其他字段
            print(f"\n=== Chunk 其他字段 ===")
            for key, value in chunk[agent_name].items():
                if key != "messages":
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        
        # 限制测试的chunk数量
        if chunk_count >= 10:
            print("\n=== 测试完成 ===")
            break

if __name__ == "__main__":
    test_stream_mode_behavior() 