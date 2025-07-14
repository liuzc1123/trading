import os
import sys

def set_api_keys():
    openai_key = ""
    finnhub_key = ""
    google_key = ""
    deepseek_key = ""
    os.environ['OPENAI_API_KEY'] = openai_key
    os.environ['FINNHUB_API_KEY'] = finnhub_key
    os.environ['GOOGLE_API_KEY'] = google_key
    os.environ['DEEPSEEK_API_KEY'] = deepseek_key
    print('API keys have been set to environment variables for this process.') 

def set_http_proxy():
    # 使用clash代理
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
    print('HTTP proxy has been set.')

if __name__ == '__main__':
    set_api_keys()
    set_http_proxy()