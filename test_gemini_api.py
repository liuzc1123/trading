import os
import google.generativeai as genai

# 设置你的API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAuAjKdBXnib9gkvTgTXhlanUqk0hGtTOs"

# 选择模型名称（如gemini-1.5-flash、gemini-1.5-pro、gemini-1.0-pro等）
MODEL_NAME = "gemini-1.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content("用一句话介绍一下人工智能是什么？")
    print("Gemini API 测试成功，返回内容：")
    print(response.text)
except Exception as e:
    print("Gemini API 测试失败，错误信息：")
    print(e) 