import json

# =================================================================
# System Instruction (System-level Instructions)
# Defines the model's persona, task boundaries, and output specifications
# =================================================================
SYSTEM_INSTRUCTION = """
You are an advanced GIS automation expert. Your task is to generate an ArcGIS Pro spatial analysis workflow that strictly conforms to JSON Schema specifications based on the natural language task description provided by the user, utilizing the provided layer information and tool list.

### Core Requirements:
1. **Tool Restriction**: You can only use tools from the provided "Tool List".
2. **Logic Coherence**: The output (output) of step S(n) must be referenced as input (input) in subsequent steps S(n+1).
3. **Coordinate System Assumption**: Unless specifically noted, assume all input data is already in projected coordinate systems.
4. **Output Format**: Output must be in valid JSON format.
5. **Quality Checks (QC)**: Each step must include reasonable `qc_checks` to verify the correctness of analysis results (for example: checking if feature counts match, if fields are empty).

### Field Naming Conventions:
- Use the format `S1.layer_name` in inputs to reference outputs from previous steps.
- Always use the format 'project.gdb/filename' in `path_hint`.
"""

# =================================================================
# User Prompt Template
# =================================================================

def get_user_prompt(task_data, toollist, schema_test):
    """
    Concatenates the task details, toollist, and Schema from task_data together
    """
    
    # Extract key information, do not show model examples in steps
    task_statement = task_data.get("task_statement", "")
    inputs_info = json.dumps(task_data.get("inputs", []), ensure_ascii=False, indent=2)
    context_info = json.dumps(task_data.get("arcgis_context", {}), ensure_ascii=False, indent=2)
    
    prompt = f"""
### 1. Task Statement (Task Description)
{task_statement}

### 2. Input Data Layers (Input Layers)
{inputs_info}

### 3. ArcGIS Runtime Context (ArcGIS Context)
{context_info}

### 4. ArcGIS Pro Tool List (Tool List)
{", ".join(toollist)}

### 5. Output Schema Constraints (Target Schema)
Please strictly follow the JSON structure below for generation:
{json.dumps(schema_test, ensure_ascii=False, indent=2)}

### 6. Start Generation:
Please return only the JSON object that conforms to the above requirements without including Markdown code block markers (such as ```json).
Additionally:
1) You can make necessary assumptions, but you must write each one into the top-level assumptions array.
2) If the task description has unclear or missing critical information, please write 1-3 questions you would ask the user in the top-level clarifying_questions array.
3) Do not stop generating the workflow due to incomplete information. Continue generating, but include uncertain points in assumptions and provide clarifying_questions.
"""
    return prompt

# =================================================================
# ArcGIS Pro Tool List
# =================================================================
ARCGIS_TOOL_LIST = [
    "Clip", "Buffer", "Spatial Join", "Intersect", 
    "Calculate Field", "Add Field", "Calculate Geometry Attributes", 
    "Kernel Density", "Feature To Point", 
    "Calculate Distance Band from Neighbor Count", 
    "Select features using attributes"
]