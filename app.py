import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import re
from io import BytesIO

# Initialize Gemini API
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
model = genai.GenerativeModel('gemini-flash-1.5')

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sample dataset
def get_sample_data():
    return {
        'sales': pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Revenue': [1000, 1200, 900, 1500, 1800, 1300],
            'Units': [100, 120, 90, 150, 180, 130],
            'Category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Electronics', 'Clothing']
        }),
        'website_traffic': pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            'Visitors': [500, 600, 450, 700, 800],
            'Bounce_Rate': [0.3, 0.25, 0.35, 0.28, 0.22]
        })
    }

def filter_data(data, filter_column=None, filter_value=None):
    """Filter DataFrame based on column and value."""
    if filter_column and filter_value:
        return data[data[filter_column] == filter_value]
    return data

def create_chart(chart_type, dataset_name, x_column, y_column, filter_column=None, filter_value=None, customization=None):
    """Create chart with customization options."""
    data = get_sample_data()[dataset_name]
    
    # Apply filtering if specified
    if filter_column and filter_value:
        data = filter_data(data, filter_column, filter_value)
    
    # Set default customization if none provided
    if customization is None:
        customization = {}
    
    # Create base chart
    if chart_type == 'bar':
        fig = px.bar(data, x=x_column, y=y_column)
    elif chart_type == 'line':
        fig = px.line(data, x=x_column, y=y_column)
    elif chart_type == 'pie':
        fig = px.pie(data, values=y_column, names=x_column)
    elif chart_type == 'scatter':
        fig = px.scatter(data, x=x_column, y=y_column)
    
    # Apply customizations
    fig.update_layout(
        title=customization.get('title', f"{y_column} by {x_column}"),
        xaxis_title=customization.get('x_axis_title', x_column),
        yaxis_title=customization.get('y_axis_title', y_column),
        template=customization.get('template', 'plotly')
    )
    
    # Update trace colors if specified
    if 'color' in customization and chart_type != 'pie':
        if chart_type == 'line':
            fig.update_traces(line_color=customization['color'])
        else:
            fig.update_traces(marker_color=customization['color'])
    
    # Update filter information in title if present
    if filter_column and filter_value:
        fig.update_layout(title=f"{fig.layout.title.text} (Filtered: {filter_column}={filter_value})")
    
    return fig

def get_chart_image(fig):
    """Convert Plotly figure to PNG image bytes."""
    return BytesIO(fig.to_image(format='png'))

def generate_chart_code(chart_type, dataset_name, x_column, y_column, filter_column=None, filter_value=None, customization=None):
    """Generate code with customization options."""
    filter_code = ""
    if filter_column and filter_value:
        filter_code = f"\n# Filter data\ndata = data[data['{filter_column}'] == '{filter_value}']"
    
    customization_code = ""
    if customization:
        customization_code = "\n# Apply customizations\nfig.update_layout(\n"
        if 'title' in customization:
            customization_code += f"    title='{customization['title']}',\n"
        if 'x_axis_title' in customization:
            customization_code += f"    xaxis_title='{customization['x_axis_title']}',\n"
        if 'y_axis_title' in customization:
            customization_code += f"    yaxis_title='{customization['y_axis_title']}',\n"
        if 'template' in customization:
            customization_code += f"    template='{customization['template']}',\n"
        customization_code += ")"
        
        if 'color' in customization and chart_type != 'pie':
            trace_update = "line_color" if chart_type == 'line' else "marker_color"
            customization_code += f"\nfig.update_traces({trace_update}='{customization['color']}')"
    
    code = f"""
import plotly.express as px

# Get data
data = get_sample_data()['{dataset_name}']{filter_code}

# Create chart
fig = px.{chart_type}(
    data,
    x='{x_column}',
    y='{y_column}'
){customization_code}

# Display chart
st.plotly_chart(fig)
"""
    return code

def extract_json_from_text(text):
    """Extract JSON object from text using regex."""
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    
    if not matches:
        return None
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def clean_llm_response(response):
    """Clean LLM response and separate code from text."""
    conversation_text = []
    code_blocks = []
    
    for part in response.parts:
        text = part.text
        
        code_pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        for match in matches:
            code_blocks.append(match.strip())
            text = text.replace(f"```{match}```", "")
            text = text.replace("```python", "")
            text = text.replace("```", "")
        
        text = text.strip()
        if text:
            conversation_text.append(text)
    
    return {
        'text': ' '.join(conversation_text),
        'code': code_blocks
    }

def get_llm_response(prompt, available_data):
    context = f"""
You are a helpful assistant that creates charts based on user requests.
Available datasets:
1. Sales data: Columns = Month, Revenue, Units, Category (Categories: Electronics, Clothing, Food)
2. Website traffic data: Columns = Day, Visitors, Bounce_Rate

Your task is to:
1. Understand what chart the user wants, including any filtering and customization requirements
2. Select appropriate dataset and columns
3. First provide a JSON response with the following structure:
{{
    "chart_type": "bar/line/pie/scatter",
    "dataset": "sales/website_traffic",
    "x_column": "column_name",
    "y_column": "column_name",
    "filter_column": "column_name_for_filtering (optional)",
    "filter_value": "value_to_filter_by (optional)",
    "customization": {{
        "title": "chart title (optional)",
        "x_axis_title": "x-axis label (optional)",
        "y_axis_title": "y-axis label (optional)",
        "color": "color name or hex code (optional)",
        "template": "plotly template name (optional)"
    }},
    "message": "Your response message to the user"
}}

Examples of valid colors: "red", "blue", "#FF0000", "rgb(255,0,0)"
Available templates: "plotly", "plotly_white", "plotly_dark", "seaborn"

Then, provide a natural language explanation of the visualization.

If you can't understand the request or it's not possible, return:
{{
    "error": "Error message explaining the issue"
}}
"""

    full_prompt = f"{context}\n\nUser request: {prompt}"

    try:
        response = model.generate_content(full_prompt)
        cleaned_response = clean_llm_response(response)
        json_data = extract_json_from_text(cleaned_response['text'])
        
        if json_data:
            json_data['conversation_text'] = cleaned_response['text'].replace(str(json_data), '').strip()
            return json_data
        else:
            return {"error": "Failed to parse the response. Please try again with a clearer request."}
            
    except Exception as e:
        return {"error": f"An error occurred while processing your request: {str(e)}"}

# [Previous helper functions (clean_llm_response, extract_json_from_text, etc.) remain the same...]

st.title("📊 AI Chart Generation Assistant")

st.markdown("""
Welcome! I'm your AI assistant for creating charts. I can help you visualize:
- Sales data (Revenue, Units by Month, filtered by Category: Electronics, Clothing, Food)
- Website traffic data (Visitors and Bounce Rate by Day)

Try asking for something like:
"Show me Electronics revenue trend over months in red color"
"Create a blue line chart of visitors with the title 'Daily Traffic Analysis'"
"Make a bar chart of monthly revenue with custom axis labels"
""")

if prompt := st.chat_input("What would you like to visualize?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    llm_response = get_llm_response(prompt, get_sample_data())
    
    if "error" in llm_response:
        st.session_state.messages.append({"role": "assistant", "content": llm_response["error"]})
    else:
        try:
            fig = create_chart(
                llm_response["chart_type"],
                llm_response["dataset"],
                llm_response["x_column"],
                llm_response["y_column"],
                llm_response.get("filter_column"),
                llm_response.get("filter_value"),
                llm_response.get("customization")
            )
            
            if "conversation_text" in llm_response:
                st.session_state.messages.append({"role": "assistant", "content": llm_response["conversation_text"]})
            else:
                st.session_state.messages.append({"role": "assistant", "content": llm_response["message"]})
            
            st.code(generate_chart_code(
                llm_response["chart_type"],
                llm_response["dataset"],
                llm_response["x_column"],
                llm_response["y_column"],
                llm_response.get("filter_column"),
                llm_response.get("filter_value"),
                llm_response.get("customization")
            ))
            
            # Display chart
            st.plotly_chart(fig)
            
            # Add PNG download button
            png_img = get_chart_image(fig)
            st.download_button(
                label="Download Chart as PNG",
                data=png_img,
                file_name="chart.png",
                mime="image/png"
            )
            
            st.session_state.messages.append({"role": "assistant", "content": "Would you like to create another chart? Feel free to ask!"})
        except Exception as e:
            error_message = f"Sorry, I encountered an error while creating the chart: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])