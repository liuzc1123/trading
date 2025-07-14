import inspect
from tradingagents.agents.utils.agent_utils import Toolkit
from tradingagents.default_config import DEFAULT_CONFIG
from set_api_keys import set_api_keys, set_http_proxy

set_api_keys()
set_http_proxy()

# 你可以在这里切换 True/False
ONLINE_MODE = True  # True: 只测线上接口，False: 只测线下接口

# 可选：动态修改 config 以影响 toolkit 行为
Toolkit._config["online_tools"] = ONLINE_MODE

class TestToolkitStaticMethods:
    def __init__(self):
        self.toolkit = Toolkit()

    def run_all_static_methods(self):
        # 获取Toolkit类的所有静态方法
        for name, member in Toolkit.__dict__.items():
            if isinstance(member, staticmethod):
                print(f"\nTesting static method: {name}")
                method = member.__func__
                try:
                    sig = inspect.signature(method)
                    kwargs = {}
                    for param in sig.parameters.values():
                        # 根据参数名和注解填充 mock 值
                        if param.name in ("ticker", "symbol", "query"):
                            kwargs[param.name] = "AAPL"
                        elif param.name in ("curr_date", "start_date", "end_date"):
                            kwargs[param.name] = "2023-12-31"
                        elif param.name == "indicator":
                            kwargs[param.name] = "rsi"
                        elif param.name == "freq":
                            kwargs[param.name] = "annual"
                        elif param.name == "look_back_days":
                            kwargs[param.name] = 7
                        elif param.annotation == int:
                            kwargs[param.name] = 7
                        elif param.annotation == str:
                            kwargs[param.name] = "test"
                        elif param.default is not inspect.Parameter.empty:
                            kwargs[param.name] = param.default
                        else:
                            kwargs[param.name] = "test"
                    result = getattr(Toolkit, name)(**kwargs)
                    if isinstance(result, str):
                        print(f"Result: {result[:500]}...\n")
                    else:
                        print(f"Result: {result}\n")
                except Exception as e:
                    print(f"Exception: {e}\n")
            else:
                print(f"{name} is not a static method")

def test_online():
    print("\n===== 测试线上接口 =====\n")
    try:
        result = Toolkit.get_YFin_data_online.invoke({"symbol": "AAPL", "start_date": "2023-12-24", "end_date": "2023-12-31"})
        print("get_YFin_data_online:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_YFin_data_online Exception:", e)
    try:
        result = Toolkit.get_stockstats_indicators_report_online.invoke({"symbol": "AAPL", "indicator": "rsi", "curr_date": "2023-12-31", "look_back_days": 7})
        print("get_stockstats_indicators_report_online:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_stockstats_indicators_report_online Exception:", e)
    try:
        result = Toolkit.get_google_news.invoke({"query": "AAPL", "curr_date": "2023-12-31"})
        print("get_google_news:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_google_news Exception:", e)
    try:
        result = Toolkit.get_social_sentiment_openai.invoke({"ticker": "AAPL", "curr_date": "2023-12-31"})
        print("get_social_sentiment_openai:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_social_sentiment_openai Exception:", e)
    try:
        result = Toolkit.get_global_news_openai.invoke({"curr_date": "2023-12-31"})
        print("get_global_news_openai:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_global_news_openai Exception:", e)
    try:
        result = Toolkit.get_fundamentals_openai.invoke({"ticker": "AAPL", "curr_date": "2023-12-31"})
        print("get_fundamentals_openai:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_fundamentals_openai Exception:", e)


def test_offline():
    print("\n===== 测试线下接口 =====\n")
    try:
        result = Toolkit.get_YFin_data.invoke({"symbol": "AAPL", "start_date": "2023-12-24", "end_date": "2023-12-31"})
        print("get_YFin_data:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_YFin_data Exception:", e)
    try:
        result = Toolkit.get_stockstats_indicators_report.invoke({"symbol": "AAPL", "indicator": "rsi", "curr_date": "2023-12-31", "look_back_days": 7})
        print("get_stockstats_indicators_report:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_stockstats_indicators_report Exception:", e)
    try:
        result = Toolkit.get_reddit_news.invoke({"curr_date": "2023-12-31"})
        print("get_reddit_news:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_reddit_news Exception:", e)
    try:
        result = Toolkit.get_finnhub_news.invoke({"ticker": "AAPL", "start_date": "2023-12-24", "end_date": "2023-12-31"})
        print("get_finnhub_news:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_finnhub_news Exception:", e)
    try:
        result = Toolkit.get_reddit_stock_info.invoke({"ticker": "AAPL", "curr_date": "2023-12-31"})
        print("get_reddit_stock_info:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_reddit_stock_info Exception:", e)
    try:
        result = Toolkit.get_finnhub_company_insider_sentiment.invoke({"ticker": "AAPL", "curr_date": "2023-12-31"})
        print("get_finnhub_company_insider_sentiment:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_finnhub_company_insider_sentiment Exception:", e)
    try:
        result = Toolkit.get_finnhub_company_insider_transactions.invoke({"ticker": "AAPL", "curr_date": "2023-12-31"})
        print("get_finnhub_company_insider_transactions:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_finnhub_company_insider_transactions Exception:", e)
    try:
        result = Toolkit.get_simfin_balance_sheet.invoke({"ticker": "AAPL", "freq": "annual", "curr_date": "2023-12-31"})
        print("get_simfin_balance_sheet:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_simfin_balance_sheet Exception:", e)
    try:
        result = Toolkit.get_simfin_cashflow.invoke({"ticker": "AAPL", "freq": "annual", "curr_date": "2023-12-31"})
        print("get_simfin_cashflow:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_simfin_cashflow Exception:", e)
    try:
        result = Toolkit.get_simfin_income_stmt.invoke({"ticker": "AAPL", "freq": "annual", "curr_date": "2023-12-31"})
        print("get_simfin_income_stmt:", result[:500], "...\n" if isinstance(result, str) else result)
    except Exception as e:
        print("get_simfin_income_stmt Exception:", e)

if __name__ == "__main__":
    if ONLINE_MODE:
        test_online()
    else:
        test_offline() 