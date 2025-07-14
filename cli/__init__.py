"""
TradingAgents CLI Package

这个包包含了TradingAgents命令行界面的所有组件：
- message_buffer: 消息缓冲区管理
- display: 用户界面显示
- user_interface: 用户交互
- analysis_runner: 分析执行
- utils: 工具函数
- models: 数据模型
"""

from .main import app

__all__ = ['app']
