# main.py
import os, json
import pandas as pd

# === Matplotlib for plot creation (headless) ===
import matplotlib
matplotlib.use("Agg")  # safe save without display
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.api.types import is_numeric_dtype

# --- 0) Load CSV ---
DF_PATH = "titanic.csv"
df = pd.read_csv(DF_PATH)

# === Plot directory + state for "last plot" ===
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
LAST_PLOT_PATH = None

def _save_current_fig(prefix: str = "plot") -> str:
    """Saves the current Matplotlib figure under ./plots/ and returns the path."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(PLOT_DIR, f"{prefix}-{ts}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()  # free memory
    return path

# --- 1) Define tools ---
from langchain_core.tools import tool

# === EDA tools ===

@tool
def tool_schema(dummy: str) -> str:
    """Returns column names and data types as JSON."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return json.dumps(schema)

@tool
def tool_nulls(dummy: str) -> str:
    """Returns columns with number of missing values as JSON (only columns with >0 missing values)."""
    nulls = df.isna().sum()
    result = {col: int(n) for col, n in nulls.items() if n > 0}
    return json.dumps(result)

@tool
def tool_describe(input_str: str) -> str:
    """
    Returns describe() statistics.
    Optional: input_str can contain a comma-separated list of columns, e.g. "age, fare".
    """
    cols = None
    if input_str and input_str.strip():
        cols = [c.strip() for c in input_str.split(",") if c.strip() in df.columns]
    stats = df[cols].describe() if cols else df.describe()
    return stats.to_csv(index=True)

# === Plot tools ===
@tool
def tool_plot_hist(params: str) -> str:
    """
    Creates a histogram for a numeric column and saves it as PNG.
    Input format: "column=age, bins=30" (bins optional, default 30).
    Return: Text incl. "PLOT:{path}" so GUI/wrapper can find the plot.
    """
    global LAST_PLOT_PATH
    # simple param parsing logic
    column, bins = None, 30
    if params:
        parts = [p.strip() for p in params.split(",")]
        for p in parts:
            if p.startswith("column="):
                column = p.split("=", 1)[1].strip()
            elif p.startswith("bins="):
                try:
                    bins = int(p.split("=", 1)[1].strip())
                except:
                    bins = 30
    if not column or column not in df.columns:
        return "Error: please provide an existing numeric column via 'column=<name>'."
    if not is_numeric_dtype(df[column]):
        return f"Error: column '{column}' is not numeric."

    series = df[column].dropna()
    if series.empty:
        return f"Error: column '{column}' has no numeric values after dropping NaNs."

    plt.figure()
    plt.hist(series, bins=bins)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Histogram of {column} (bins={bins})")
    path = _save_current_fig(prefix=f"hist-{column}")
    LAST_PLOT_PATH = path
    return f"Histogram created for '{column}'. PLOT:{path}"

@tool
def tool_plot_scatter(params: str) -> str:
    """
    Creates a scatter plot for two numeric columns (x vs. y) and saves it as PNG.
    Input format: "x=age, y=fare"
    Optional: ", sample=1000" to sample n rows.
    """
    global LAST_PLOT_PATH
    x = y = None
    sample = None
    if params:
        parts = [p.strip() for p in params.split(",")]
        for p in parts:
            if p.startswith("x="):
                x = p.split("=", 1)[1].strip()
            elif p.startswith("y="):
                y = p.split("=", 1)[1].strip()
            elif p.startswith("sample="):
                try:
                    sample = int(p.split("=", 1)[1].strip())
                except:
                    sample = None
    if not x or not y or x not in df.columns or y not in df.columns:
        return "Error: provide existing numeric columns via 'x=<name>, y=<name>'."
    if not (is_numeric_dtype(df[x]) and is_numeric_dtype(df[y])):
        return f"Error: both '{x}' and '{y}' must be numeric."

    data = df[[x, y]].dropna()
    if sample and sample < len(data):
        data = data.sample(sample, random_state=42)

    plt.figure()
    plt.scatter(data[x], data[y], s=12, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter: {x} vs. {y} (n={len(data)})")
    path = _save_current_fig(prefix=f"scatter-{x}-vs-{y}")
    LAST_PLOT_PATH = path
    return f"Scatter plot created for '{x}' vs '{y}'. PLOT:{path}"

@tool
def tool_plot_bar_counts(params: str) -> str:
    """
    Creates a bar chart of value counts for a categorical column.
    Input format: "column=sex, top=10" (top optional, default 10).
    """
    global LAST_PLOT_PATH
    column, top = None, 10
    if params:
        parts = [p.strip() for p in params.split(",")]
        for p in parts:
            if p.startswith("column="):
                column = p.split("=", 1)[1].strip()
            elif p.startswith("top="):
                try:
                    top = int(p.split("=", 1)[1].strip())
                except:
                    top = 10
    if not column or column not in df.columns:
        return "Error: provide an existing column via 'column=<name>'."

    counts = df[column].dropna().astype(str).value_counts().head(top)
    if counts.empty:
        return f"Error: column '{column}' has no values."

    plt.figure(figsize=(6, 3.8))
    counts.iloc[::-1].plot(kind="barh")  # reversed for nicer top-down order
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.title(f"Top {len(counts)} '{column}' values")
    path = _save_current_fig(prefix=f"bar-{column}")
    LAST_PLOT_PATH = path
    return f"Bar chart created for '{column}'. PLOT:{path}"

# --- 2) Wire up tools for LangChain ---
tools = [
    tool_schema,
    tool_nulls,
    tool_describe,
    # === Add plot tools ===
    tool_plot_hist,
    tool_plot_scatter,
    tool_plot_bar_counts,
]

# --- 3) Configure LLM ---
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
else:
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# --- 4) Narrow policy/prompt (agent behavior) ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = (
    "You are a data-focused assistant. "
    "If a question requires information from the CSV, first use an appropriate tool. "
    "Use only one tool call per step if possible. "
    "Answer concisely and in a structured way. "
    "If no tool fits, briefly explain why.\n\n"
    "You can create visualizations if needed (histogram, scatter, bar chart). "
    "For plots, ONLY use the plot tools and stick to their input formats.\n\n"
    "Available tools:\n{tools}\n"
    "Use only these tools: {tool_names}."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

_tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
_tool_names = ", ".join(t.name for t in tools)
prompt = prompt.partial(tools=_tool_desc, tool_names=_tool_names)

# --- 5) Create & run tool-calling agent ---
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,   # optional: True for debug logs
    max_iterations=3,
)

# --- Public API ---
def ask_agent(query: str) -> str:
    """Backward-compatible: text-only answer (for mini_eval etc.)."""
    # Plot state not needed
    return agent_executor.invoke({"input": query})["output"]

def ask_agent_with_plot(query: str):
    """
    Returns (text, plot_path_or_none).
    Plot tools set LAST_PLOT_PATH internally and return 'PLOT:<path>' in the text.
    """
    global LAST_PLOT_PATH
    LAST_PLOT_PATH = None
    out = agent_executor.invoke({"input": query})["output"]
    return out, LAST_PLOT_PATH

if __name__ == "__main__":
    # Example that will definitely trigger a plot:
    demo_q = "Create a histogram of the column age with 20 bins. Use 'column=age, bins=20'."
    text, plot = ask_agent_with_plot(demo_q)
    print("\n=== AGENT ANSWER ===")
    print(text)
    if plot:
        print(f"(Plot saved under: {plot})")
