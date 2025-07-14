#!/usr/bin/env python3
"""
直接调试CLI分析函数的脚本
绕过typer命令行参数，直接调用run_analysis函数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置API密钥
from set_api_keys import set_api_keys, set_http_proxy
set_api_keys()
set_http_proxy()

# 导入CLI相关模块
from cli.main import run_analysis

def debug_cli_analysis():
    """直接调用CLI分析函数进行调试"""
    print("=== 开始直接调试CLI分析 ===")
    
    # 🔴 在这里设置断点！这是CLI分析的入口点
    # 当程序在这里停止时，您可以进入run_analysis函数进行调试
    
    try:
        # 直接调用分析函数
        # 这会启动完整的用户交互流程
        run_analysis()
        
    except KeyboardInterrupt:
        print("\n用户中断了分析过程")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_cli_analysis() 