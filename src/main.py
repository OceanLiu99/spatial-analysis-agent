from google import genai
from google.genai import types

import json
import os
import pandas as pd
from dotenv import load_dotenv
from prompts import SYSTEM_INSTRUCTION, get_user_prompt, ARCGIS_TOOL_LIST



# set up Gemini API client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)

def run_all_experiments(task_ids=None):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Set base path to project root
    tasks_csv_path = os.path.join(base_path, "data/tasks/tasks_test.csv")
    schema_path = os.path.join(base_path, "src/schema.json")
    output_dir = os.path.join(base_path, "data/pred")

    # Read tasks from CSV
    tasks_df = pd.read_csv(tasks_csv_path)
    all_tasks = tasks_df.to_dict(orient="records")

    # Filter tasks to only include the specified task IDs (if provided)
    if task_ids:
        all_tasks = [task for task in all_tasks if task.get("task_id") in task_ids]

    # Load schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_test = json.load(f)
    
    for task_data in all_tasks:
        task_id = task_data.get("task_id", "unknown")  # Default to "unknown" if task_id is missing
        task_statement = task_data.get("task_statement", "No task statement provided")  # Ensure task_statement is read
        
        # combined prompt: task data + layer info + tool list + schema
        final_prompt = get_user_prompt(task_data, ARCGIS_TOOL_LIST, schema_test)

        
        try:
            print(f"Processing Task {task_id}: {task_statement}")
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            prediction = json.loads(response.text)
            
            # save prediction to file
            output_file = os.path.join(output_dir, f"pred_task_{task_id}.json")
            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(prediction, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
    
if __name__ == "__main__":
    run_all_experiments(task_ids=list(range(2, 10)))