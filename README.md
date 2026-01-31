# 🛰️ Spatial Analysis Workflow Agent

An intelligent GIS automation system that leverages AI (Google Gemini) to automatically generate ArcGIS Pro spatial analysis workflows from natural language task descriptions.

## Overview

**Spatial Analysis Workflow Agent** transforms user-input spatial analysis task descriptions into structured ArcGIS Pro workflows. By combining natural language processing with GIS expertise, this system enables both GIS professionals and non-specialists to create complex spatial analysis pipelines with high efficiency.

### Key Features

- **Natural Language Interface**: Describe your spatial analysis tasks in plain English
- **AI-Powered Workflow Generation**: Automatically generates valid ArcGIS Pro workflows using Google Gemini
- **Shapefile Support**: Upload and parse geographic datasets in shapefile format
- **Structured Output**: Generates workflows in both plain steps summary and strict JSON Schema format
- **Quality Checks**: Includes automated validation rules for each workflow step
- **Web Interface**: Interactive Streamlit-based UI for easy access and testing
- **Model Evaluation**: Score and benchmark AI accuracy to compare and select the best performing model

## Project Structure

```
spatial-analysis-agent/
├── src/
│   ├── streamlit_app.py          # Main web interface
│   ├── prompts.py                # Gemini prompt templates and system instructions
│   ├── schema.json               # JSON Schema for workflow validation
│   ├── main.py                   # Main entry point
│   ├── server.py                 # API server implementation
│   ├── gemini_test.py            # Gemini API testing utilities
│   ├── agent/
│   │   └── data_catalog_dbf.py   # DBF file catalog utilities
│   ├── eval/
│   │   └── auto_score.py         # Evaluation and scoring module
│   └── validation/               # Validation utilities
├── data/
│   ├── tasks/                    # Sample task descriptions
│   ├── pred/                     # Prediction workflows results
│   ├── truth/                    # Ground truth workflows
│   └── reports/                  # Evaluation reports
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API keys)
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment manager (venv or conda)
- Google Gemini API key

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spatial-analysis-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv-spatial
   source .venv-spatial/bin/activate  # On Windows: .venv-spatial\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Web Interface (Streamlit)

Launch the interactive web application:

```bash
cd src
streamlit run streamlit_app.py
```

The app will open in your browser (typically at `http://localhost:8501`).

#### Workflow:

1. **Data Tab**
   - Upload a shapefile archive (.zip containing .shp, .shx, .dbf, .prj files)
   - System automatically extracts and displays layer fields
   - View sample attribute rows

![upload_data_bundle](https://github.com/user-attachments/assets/b7087052-e3cc-4427-92cd-3efbfe440396)

2. **Agent Tab**
   - Describe your spatial analysis task in natural language
   - Specify layer name (optional)
   - Click "Generate Workflow" to create the workflow
   - Review workflow as text or JSON
   - Download the generated workflow as `workflow.json`
![workflow_generated](https://github.com/user-attachments/assets/3958a4f1-2c73-44cc-ac46-76004433a1bb)
![workflow_json](https://github.com/user-attachments/assets/aeb28eb8-8bc7-4c71-b663-c07f265eb0bd)


## Supported ArcGIS Tools

The system currently supports the following ArcGIS Pro tools:

- Clip
- Buffer
- Spatial Join
- Intersect
- Calculate Field
- Add Field
- Calculate Geometry Attributes
- Kernel Density
- Feature To Point
- Calculate Distance Band from Neighbor Count
- Select features using attributes

Tools can be extended by modifying `ARCGIS_TOOL_LIST` in `prompts.py`.

## Core Concepts

### System Instructions

The Gemini model operates under strict guidelines defined in `SYSTEM_INSTRUCTION`:

1. **Tool Restriction**: Only uses tools from the list
2. **Logic Coherence**: Output of step S(n) becomes input for step S(n+1)
3. **Coordinate System Handling**: Assumes projected coordinate systems by default
4. **JSON Compliance**: All outputs must be valid JSON
5. **Quality Assurance**: Each step includes quality checks

### Assumptions & Clarifying Questions

When task descriptions are incomplete:
- The system documents assumptions in the `assumptions` array
- It generates clarifying questions in the `clarifying_questions` array
- It continues generating workflows rather than failing

## Development

### Testing

Run the test suite:

```bash
python src/gemini_test.py
```

### Evaluation

Evaluate workflow generation quality:

```bash
python src/eval/auto_score.py
```

### API Endpoints

If running the server:

```bash
python src/server.py
```

Refer to `server.py` for available endpoints.

### Sample Data Structure

- `data/tasks/`: Sample task definitions
- `data/truth/`: Ground truth workflows for validation
- `data/pred/`: Model-generated predictions for evaluation
- `data/reports/`: Analysis reports and metrics

## Future Work

### Future Enhancements

- [ ] Multi-step workflow validation and execution
- [ ] Integration with ArcGIS Online
- [ ] Support for more GIS tools and operations

## License

See [LICENSE](LICENSE) file for details.

---

**Last Updated**: January 2026  
**Version**: 1.0.0
