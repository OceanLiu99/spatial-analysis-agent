import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from prompts import SYSTEM_INSTRUCTION, get_user_prompt, ARCGIS_TOOL_LIST

# ---------- Gemini client setup ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
schema_test = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

# ---------- FastAPI app ----------
app = FastAPI(title="Spatial Analysis Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request/response models ----------
class FieldInfo(BaseModel):
    name: str
    type: str = ""
    sample: str | None = None

class GenerateRequest(BaseModel):
    task_statement: str = Field(..., min_length=6)
    layer_name: str = "UploadedLayer"
    fields: List[FieldInfo] = Field(default_factory=list)
    arcgis_context: dict = Field(default_factory=dict)

class GenerateResponse(BaseModel):
    workflow_json: dict
    workflow_text: str

# ---------- Routes ----------
@app.post("/api/generate-workflow", response_model=GenerateResponse)
async def generate_workflow(payload: GenerateRequest) -> GenerateResponse:
    task_data = {
        "task_id": "ui_demo",
        "task_statement": payload.task_statement,
        "inputs": [
            {
                "layer_name": payload.layer_name,
                "geometry_type": "",
                "required_fields": [f.name for f in payload.fields],
                "optional_fields": [],
                "notes": ""
            }
        ],
        "arcgis_context": payload.arcgis_context,
    }

    prompt = get_user_prompt(task_data, ARCGIS_TOOL_LIST, schema_test)

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        prediction = json.loads(response.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini request failed: {exc}") from exc

    readable = []
    for step in prediction.get("steps", []):
        readable.append(
            f"{step.get('step_id','')} {step.get('tool','')}\n"
            f"- purpose: {step.get('purpose','')}\n"
            f"- inputs: {', '.join(i.get('layer_name','') for i in step.get('inputs', []))}\n"
            f"- outputs: {', '.join(o.get('path_hint','') for o in step.get('outputs', []))}"
        )

    return GenerateResponse(workflow_json=prediction, workflow_text="\n\n".join(readable))
