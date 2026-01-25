from google.genai import Client
from google.genai import types
import json
import os
from dotenv import load_dotenv
from prompts import SYSTEM_INSTRUCTION, get_user_prompt, ARCGIS_TOOL_WHITELIST

# 1. 加载环境变量
load_dotenv(dotenv_path=".env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("未在 .env 文件中找到 GOOGLE_API_KEY")

# 2. 初始化新版 Client
client = Client(api_key=api_key)

def run_experiment(task_index):
    base_path = "src/spatial_agent"
    task_file = f"{base_path}/example_task{task_index}.json"
    schema_file = f"{base_path}/schema_test.json"
    output_file = f"{base_path}/pred_task{task_index}.json"

    with open(task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_test = json.load(f)

    final_prompt = get_user_prompt(task_data, ARCGIS_TOOL_WHITELIST, schema_test)
    
    try:
        # 3. 使用新版 generate_content 语法
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=final_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json"
            )
        )
        
        # 新版解析方式：response.parsed 或是 response.text
        # 由于指定了 mime_type，直接 json.loads(response.text) 即可
        prediction = json.loads(response.text)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Task {task_index} 处理成功")

    except Exception as e:
        print(f"❌ Task {task_index} 处理失败: {str(e)}")

if __name__ == "__main__":
    for i in range(1, 11):
        run_experiment(i)