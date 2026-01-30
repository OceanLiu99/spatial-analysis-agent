import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import shapefile  # pyshp
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import SYSTEM_INSTRUCTION, get_user_prompt, ARCGIS_TOOL_LIST


# ---------- CACHED RESOURCES ----------
@st.cache_resource
def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before launching Streamlit")
    return genai.Client(api_key=api_key)


@st.cache_data
def load_schema() -> Dict:
    schema_path = Path(__file__).resolve().parent / "schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


# ---------- HELPERS ----------
def extract_fields_from_zip(uploaded_file) -> Tuple[List[Dict], List[Dict]]:
    """Return (fields, sample_rows) extracted from the first DBF in the zip."""
    file_bytes = uploaded_file.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
            zf.extractall(tmpdir_path)
        dbfs = sorted(
            p
            for p in tmpdir_path.rglob("*.dbf")
            if "__MACOSX" not in p.parts and not p.name.startswith("._") and p.stat().st_size > 0
        )
        if not dbfs:
            raise FileNotFoundError("压缩包内未找到 .dbf 文件")
        dbf_path = max(dbfs, key=lambda p: p.stat().st_size)
        reader = shapefile.Reader(str(dbf_path))
        raw_fields = reader.fields[1:]
        field_defs = []
        for name, ftype, size, dec in raw_fields:
            field_defs.append({
                "name": str(name),
                "type": str(ftype),
                "size": int(size),
                "decimal": int(dec),
            })
        samples = []
        for rec in reader.iterRecords():
            entry = {field_defs[i]["name"]: rec[i] for i in range(min(len(field_defs), len(rec)))}
            samples.append(entry)
            if len(samples) >= 3:
                break
        return field_defs, samples


def call_gemini(task_statement: str, layer_name: str, fields: List[Dict]) -> Dict:
    client = get_gemini_client()
    schema_test = load_schema()
    task_data = {
        "task_id": "streamlit_ui",
        "task_statement": task_statement,
        "inputs": [
            {
                "layer_name": layer_name or "UploadedLayer",
                "geometry_type": "",
                "required_fields": [f["name"] for f in fields],
                "optional_fields": [],
                "notes": ""
            }
        ],
        "arcgis_context": {}
    }
    prompt = get_user_prompt(task_data, ARCGIS_TOOL_LIST, schema_test)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    return json.loads(response.text)


def flatten_workflow_text(prediction: Dict) -> str:
    parts = []
    for step in prediction.get("steps", []):
        step_id = step.get("step_id", "")
        tool = step.get("tool", "")
        purpose = step.get("purpose", "")
        inputs = ", ".join(inp.get("layer_name", "") or inp.get("source", "") for inp in step.get("inputs", []))
        outputs = ", ".join(out.get("path_hint", "") for out in step.get("outputs", []))
        parts.append(f"{step_id} · {tool}\npurpose: {purpose}\ninput: {inputs}\noutput: {outputs}")
    return "\n\n".join(parts) if parts else "(model returned no steps)"


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Spatial Analysis Agent", layout="wide")

if "fields" not in st.session_state:
    st.session_state["fields"] = []
if "samples" not in st.session_state:
    st.session_state["samples"] = []
if "workflow" not in st.session_state:
    st.session_state["workflow"] = None

# Header with title
st.markdown(
    """
    <style>
    .header-container {
        padding: 12px 0;
        border-bottom: 1px solid #e0e0e0;
    }
    .header-title {
        font-size: 32px;
        font-weight: bold;
        color: #1f2937;
    }
    </style>
    <div class="header-container">
        <div class="header-title">🛰️ Spatial Analysis Workflow Agent</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main layout: Steps bar on the left, content on the right
st.markdown(
    """
    <style>
    .steps-column {
        padding-right: 20px;
        border-right: 1px solid #e0e0e0;
    }
    .steps-title {
        font-weight: bold;
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 10px;
        font-size: 14px;
        font-weight: 500;
        width: 100%;
        margin-bottom: 10px;
        color: #333;
    }
    .stButton button:hover {
        background-color: #f9f9f9;
        border-color: #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([0.3, 2.7], gap="large")

# Left column: Steps bar
with col1:
    st.markdown('<div class="steps-column">', unsafe_allow_html=True)
    st.markdown('<div class="steps-title">STEPS</div>', unsafe_allow_html=True)
    if st.button("Data", key="data_button", use_container_width=True):
        st.session_state["section"] = "Data"
    if st.button("Agent", key="agent_button", use_container_width=True):
        st.session_state["section"] = "Agent"
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize section if not set
if "section" not in st.session_state:
    st.session_state["section"] = "Data"

section = st.session_state["section"]

# Right column: Content
with col2:
    if section == "Data":
        st.subheader("Step 1 · Upload Data Bundle")
        uploaded = st.file_uploader("Shapefile archive (.zip)", type=["zip"], help="Include .shp/.shx/.dbf/.prj")
        if uploaded:
            try:
                with st.spinner("Parsing fields..."):
                    fields, samples = extract_fields_from_zip(uploaded)
                st.success(f"Parsed {len(fields)} fields")
                st.session_state["fields"] = fields
                st.session_state["samples"] = samples
                st.dataframe(fields, use_container_width=True)
                if samples:
                    st.write("Sample attribute rows:")
                    st.json(samples, expanded=False)
            except Exception as exc:
                st.error(f"Failed to parse: {exc}")
    else:
        st.subheader("Step 2 · Describe the Task & Build Workflow")
        if not st.session_state["fields"]:
            st.warning("Upload a zip on the Data panel to extract fields first.")
        task_statement = st.text_area("Analysis task description", height=180)
        layer_name = st.text_input("Layer name", value="UploadedLayer")
        if st.button("Generate Workflow", type="primary", disabled=not task_statement):
            try:
                with st.spinner("Calling Gemini..."):
                    prediction = call_gemini(task_statement, layer_name, st.session_state["fields"])
                st.session_state["workflow"] = prediction
            except Exception as exc:
                st.session_state["workflow"] = None
                st.error(f"Failed to generate: {exc}")
        if st.session_state.get("workflow"):
            prediction = st.session_state["workflow"]
            st.success("Workflow generated")
            st.markdown("**Text summary**")
            st.code(flatten_workflow_text(prediction), language="text")
            st.markdown("**JSON**")
            st.json(prediction)
            st.download_button(
                "Download workflow.json",
                data=json.dumps(prediction, ensure_ascii=False, indent=2),
                file_name="workflow.json",
                mime="application/json"
            )
