# app.py
import streamlit as st
from main import ask_agent_with_plot

st.set_page_config(page_title="CSV Plot Agent", layout="centered")
st.title("CSV Plot Agent (LangChain + Streamlit)")

st.info("""
**How to use this app**
- Use the Quick Check buttons for instant info about the CSV (Schema, Missing Values, Describe).
- Or enter your own query below. You can ask for text answers or generate plots (Histogram, Scatter, Bar chart). Use the provided examples for best results.
""")

# Predefined quick buttons (useful for demo)
col1, col2, col3 = st.columns(3)
if col1.button("Columns & Data Types"):
    q = "Show me the schema (columns + data types). Use tool_schema."
    text, plot = ask_agent_with_plot(q)
    st.write(text)
if col2.button("Check Missing Values"):
    q = "Which columns have missing values? List 'Column: Count'."
    text, plot = ask_agent_with_plot(q)
    st.write(text)
if col3.button("Describe Numeric Columns"):
    q = "Give me a statistical summary of the numeric columns."
    text, plot = ask_agent_with_plot(q)
    st.write(text)

st.divider()

# Free input
query = st.text_input("Ask the agent anything (e.g. 'Create a histogram of age with 25 bins.')")
help_expander = st.expander("Help & Examples")
with help_expander:
    st.markdown("""
### What you can ask
The agent can only use the tools you see here:
- **Schema** → Columns & data types  
- **Missing Values** → Columns with NaNs and their counts  
- **Describe** → Summary stats (mean, min, max, etc.) for numeric columns  
- **Plots** → Histogram, Scatter plot, Bar chart  

### Plot examples (exact format helps the agent)
- Histogram: `Create a histogram. Use: column=age, bins=25.`
- Scatter: `Create a scatter plot: x=age, y=fare, sample=500.`
- Bar (value_counts): `Bar chart of the most frequent categories: column=sex, top=5.`

### Text-only examples
- `Show me the first 3 columns with data types.`
- `Which columns have missing values?`
- `Give me describe() for age and fare.`

### Important notes
- The agent cannot do everything (only what the defined tools allow).  
  For example, sorting rows or training ML models will not work.  
- For plots, please stick to the input formats above (`column=...`, `x=..., y=...`).  
- Answers are short and structured (this is not ChatGPT, it’s a data agent working with a CSV).
""")


if st.button("Run Agent", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agent is thinking and calling tools..."):
            text, plot = ask_agent_with_plot(query)
        st.subheader("Answer")
        st.write(text)
        if plot:
            st.subheader("Plot")
            st.image(plot, use_column_width=True)
