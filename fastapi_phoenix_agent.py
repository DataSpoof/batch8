"""
FastAPI app wrapping the provided Phoenix + OpenAI agent tooling.
Save as `fastapi_phoenix_agent.py` and run with:

    pip install fastapi uvicorn openai python-dotenv pandas duckdb pydantic arize-phoenix openinference-instrumentation-openai
    uvicorn fastapi_phoenix_agent:app --reload --port 8000

Notes:
- This file expects environment variables OPENAI_API_KEY and PHOENIX_TRACES_ENDPOINT (optional).
- For local Phoenix collector you can set PHOENIX_TRACES_ENDPOINT to e.g. http://localhost:6006/v1/traces
- The code re-uses functions from your snippet but exposes them as HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import os
import json
import pandas as pd
import duckdb
import warnings

# OpenTelemetry / Phoenix / OpenAI imports
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import StatusCode
from openinference.instrumentation import TracerProvider
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

warnings.filterwarnings('ignore')

# ---------- Configuration ----------
TRANSACTION_DATA_FILE_PATH ="Sales.parquet"
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PHOENIX_TRACES_ENDPOINT =os.getenv("PHONENIX_ENDPOINT") 
PHOENIX_GRPC_PORT ="4318"
PROJECT_NAME = "default"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")

# ---------- Initialize OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Initialize Phoenix tracer provider ----------
os.environ["PHOENIX_GRPC_PORT"] = PHOENIX_GRPC_PORT
if "PHOENIX_COLLECTOR_ENDPOINT" in os.environ:
    del os.environ["PHOENIX_COLLECTOR_ENDPOINT"]

# Attempt to launch Phoenix UI locally if available (best-effort)
try:
    session = px.launch_app()
    phoenix_ui_url = getattr(session, "url", None)
except Exception:
    phoenix_ui_url = None

tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=PHOENIX_TRACES_ENDPOINT,
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)

# ---------- Prompts from original snippet ----------
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

# ---------- Helper functions (same logic, adapted for server) ----------

def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt,
        columns=", ".join(map(str, columns)),
        table_name=table_name,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    return response.choices[0].message.content


def lookup_sales_data(prompt: str) -> str:
    """Read parquet, ask LLM to generate SQL, execute it via DuckDB, return CSV/text result."""
    try:
        table_name = "sales"
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        sql_query = generate_sql_query(prompt, df.columns.tolist(), table_name)
        sql_query = sql_query.strip().replace("```sql", "").replace("```", "")

        with tracer.start_as_current_span("execute_sql_query", openinference_span_kind="chain") as span:
            try:
                span.set_attribute("db.system", "duckdb")
                span.set_attribute("db.statement", sql_query)
            except Exception:
                pass

            result = duckdb.sql(sql_query).df()
            span.set_attribute("result.rows", int(result.shape[0]))
            span.set_status(StatusCode.OK)
            span.set_attribute("result.preview", result.head(5).to_dict())

        return result.to_json(orient="records")
    except Exception as e:
        with tracer.start_as_current_span("execute_sql_query_error", openinference_span_kind="chain") as err_span:
            err_span.set_status(StatusCode.ERROR)
            err_span.set_attribute("error.message", str(e))
        raise


def analyze_sales_data(prompt: str, data: str) -> str:
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


from pydantic import BaseModel as PydanticBaseModel, Field
class VisualizationConfig(PydanticBaseModel):
    chart_type: str = Field(...)
    x_axis: str = Field(...)
    y_axis: str = Field(...)
    title: str = Field(...)


def extract_chart_config(data: str, visualization_goal: str) -> dict:
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data, visualization_goal=visualization_goal)
    try:
        # prefer structured parse; if SDK doesn't support parse in user's env, fallback
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_format=VisualizationConfig,
        )
        content = response.choices[0].message.content
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data,
        }
    except Exception:
        return {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data,
        }


def create_chart(config: dict) -> str:
    formatted_prompt = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
""".format(config=json.dumps(config, default=str))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def generate_visualization(data: str, visualization_goal: str) -> str:
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code

# ---------- FastAPI app and request models ----------
app = FastAPI(title="Phoenix + OpenAI Agent API")

class LookupRequest(BaseModel):
    prompt: str

class AnalyzeRequest(BaseModel):
    prompt: str
    data: str

class VisualizeRequest(BaseModel):
    data: str
    visualization_goal: str

class AgentMessage(BaseModel):
    role: str
    content: Any

class AgentRunRequest(BaseModel):
    messages: List[AgentMessage]


@app.get("/")
def root():
    return {
        "status": "ok",
        "phoenix_ui": phoenix_ui_url,
        "phoenix_traces_endpoint": PHOENIX_TRACES_ENDPOINT,
    }


@app.post("/lookup")
def api_lookup(req: LookupRequest):
    try:
        result_json = lookup_sales_data(req.prompt)
        return {"result": json.loads(result_json)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def api_analyze(req: AnalyzeRequest):
    try:
        analysis = analyze_sales_data(req.prompt, req.data)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
def api_visualize(req: VisualizeRequest):
    try:
        code = generate_visualization(req.data, req.visualization_goal)
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Define tools/functions that can be called by the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data",
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["data", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "visualization_goal": {"type": "string", "description": "The goal of the visualization."}
                },
                "required": ["data", "visualization_goal"]
            }
        }
    }
]

# Dictionary mapping function names to their implementations
tool_implementations = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data,
    "generate_visualization": generate_visualization
}

# code for executing the tools returned in the model's response
def handle_tool_calls(tool_calls, messages):

    for tool_call in tool_calls:
        function = tool_implementations[tool_call.function.name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})

    return messages

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
"""


def run_agent(messages):
    print("Running agent with messages:", messages)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # Check and add system prompt if needed
    if not any(
            isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
            messages.append(system_prompt)

    while True:
        print("Making router call to OpenAI")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        messages.append(response.choices[0].message)
        tool_calls = response.choices[0].message.tool_calls
        print("Received response with tool calls:", bool(tool_calls))

        # if the model decides to call function(s), call handle_tool_calls
        if tool_calls:
            print("Processing tool calls")
            messages = handle_tool_calls(tool_calls, messages)
        else:
            print("No tool calls, returning final response")
            return response.choices[0].message.content

from fastapi import HTTPException
from typing import Any, Dict, List

@app.post("/agent/run")
def api_agent_run(req: AgentRunRequest):
    """
    Execute the sales agent with the provided chat messages.
    Ensures a system prompt is present (inserted at index 0 if missing)
    and returns the agent's final text response.
    """
    try:
        # Normalize to a plain list[dict] for the OpenAI client
        raw_messages: List[Dict[str, Any]] = []
        for m in req.messages:
            # Accept both Pydantic models and plain dicts
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
                if not role or content is None:
                    raise HTTPException(status_code=422, detail="Each message must include 'role' and 'content'.")
                raw_messages.append({"role": role, "content": content})
            else:
                md = m.model_dump() if hasattr(m, "model_dump") else m.dict()
                role = md.get("role")
                content = md.get("content")
                if not role or content is None:
                    raise HTTPException(status_code=422, detail="Each message must include 'role' and 'content'.")
                raw_messages.append({"role": role, "content": content})

        # Insert the system prompt first if not present
        if not any((isinstance(msg, dict) and msg.get("role") == "system") for msg in raw_messages):
            raw_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # Run the agent loop (this will also handle tool calls)
        result_text = run_agent(raw_messages)

        return {
            "ok": True,
            "result": result_text,
        }

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions untouched
        raise
    except Exception as e:
        # Log if you have a logger; keep response clean for clients
        # logger.exception("api_agent_run failed")  # optional
        raise HTTPException(status_code=500, detail=f"Agent run failed: {str(e)}")

# If run directly, start uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_phoenix_agent:app", host="0.0.0.0", port=8000, reload=True)
