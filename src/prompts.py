import json

# =================================================================
# System Instruction (系统级指令)
# 定义模型的人设、任务边界和输出规范
# =================================================================
SYSTEM_INSTRUCTION = """
你是一个高级 GIS 自动化专家。你的任务是根据用户输入的自然语言任务描述，利用提供的图层信息和工具白名单，
生成一个符合严格 JSON Schema 规范的 ArcGIS Pro 空间分析工作流（Workflow）。

### 核心规范：
1. **工具限制**：只能使用提供的“工具名单”中的工具。
2. **逻辑连贯**：步骤 S(n) 的输出（output）必须作为后续步骤 S(n+1) 的输入（input）进行引用。
3. **坐标系假设**：除非特殊说明，假设所有输入数据均已处于投影坐标系。
4. **输出格式**：输出必须是合法的 JSON 格式。
5. **质量检查 (QC)**：每个步骤必须包含合理的 `qc_checks`，以验证分析结果的正确性（例如：检查要素数量是否匹配、字段是否为空）。

### 字段命名约定：
- 使用 `S1.layer_name` 的形式在 inputs 中引用上一个步骤的输出。
- 在 `path_hint` 中始终使用 'project.gdb/文件名' 的格式。
"""

# =================================================================
# User Prompt Template
# =================================================================

def get_user_prompt(task_data, whitelist, schema_test):
    """
    将 task_data 中的任务详情、白名单和 Schema 拼接到一起
    """
    
    # 提取关键信息，不给模型看 example 中的 steps
    task_statement = task_data.get("task_statement", "")
    inputs_info = json.dumps(task_data.get("inputs", []), ensure_ascii=False, indent=2)
    context_info = json.dumps(task_data.get("arcgis_context", {}), ensure_ascii=False, indent=2)
    
    prompt = f"""
### 1. 任务描述 (Task Statement)
{task_statement}

### 2. 输入数据图层 (Input Layers)
{inputs_info}

### 3. ArcGIS 运行上下文 (ArcGIS Context)
{context_info}

### 4. ArcGIS Pro 工具白名单 (Tool Whitelist)
{", ".join(whitelist)}

### 5. 输出 Schema 约束 (Target Schema)
请严格参考以下 JSON 结构进行生成：
{json.dumps(schema_test, ensure_ascii=False, indent=2)}

### 6. 开始生成：
请直接返回符合上述要求的 JSON 对象，不要包含 Markdown 代码块标记（如 ```json ）。
另外：
1) 你可以做必要的 assumptions，但必须把每一条写进顶层 assumptions 数组。
2) 如果任务描述里有不清楚或缺失的关键信息，请在顶层 clarifying_questions 数组里写出 1-3 个你会问用户的问题。
3) 不要因为信息不全而停止生成 workflow。继续生成，但把不确定点写进 assumptions，并提出 clarifying_questions。
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