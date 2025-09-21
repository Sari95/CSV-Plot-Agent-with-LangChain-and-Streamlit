# CSV-Plot-Agent-with-LangChain-and-Streamlit
CSV Plot Agent with LangChain and Streamlit: Learning project - introduction to Data Agents

This project shows how to build a simple CSV Plot Agent using **LangChain**, **GPT-4o-mini (or Ollama)**, and **Streamlit**.  
The agent can answer questions about a CSV dataset in natural language and generate quick plots (histogram, scatter, bar chart).

## Features
- Quick Checks (Schema, Missing Values, Describe)
- Natural language queries with text answers or plots
- Streamlit GUI for easy interaction

## Getting Started
```bash
# Clone repo and navigate into folder
git clone <your-repo-url>
cd csv-plot-agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux

# Install packages
pip install "langchain>=0.2,<0.3" "langchain-openai>=0.1.7" "langchain-community>=0.2" pandas seaborn matplotlib streamlit

# Run app
streamlit run app.py
