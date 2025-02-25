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
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0 # Add counter for unique charts
if 'download_counter' not in st.session_state:
    st.session_state.download_counter = 0  # Add counter for unique download buttons


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

def get_llm_response(prompt, available_data):
    response_schema = {
        "type": "object",
        "properties": {
            "is_error": {
                "type": "boolean",
                "description": "Flag indicating if the response is an error"
            },
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie", "scatter"],
                "description": "Type of chart to generate"
            },
            "dataset": {
                "type": "string",
                "enum": ["sales", "website_traffic"],
                "description": "Dataset to use for the chart"
            },
            "x_column": {
                "type": "string",
                "description": "Column name for x-axis"
            },
            "y_column": {
                "type": "string",
                "description": "Column name for y-axis"
            },
            "filter_column": {
                "type": "string",
                "description": "Column name for filtering (optional)"
            },
            "filter_value": {
                "type": "string",
                "description": "Value to filter by (optional)"
            },
            "customization": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Chart title"
                    },
                    "x_axis_title": {
                        "type": "string",
                        "description": "X-axis label"
                    },
                    "y_axis_title": {
                        "type": "string",
                        "description": "Y-axis label"
                    },
                    "color": {
                        "type": "string",
                        "description": "Color for the chart (name or hex code)"
                    },
                    "template": {
                        "type": "string",
                        "enum": ["plotly", "plotly_white", "plotly_dark", "seaborn"],
                        "description": "Chart template name"
                    }
                }
            },
            "message": {
                "type": "string",
                "description": "Response message to the user"
            }
        },
        "required": ["is_error", "message"]
        
    }
    
    context = f"""
You are a helpful assistant that ONLY creates charts based on user requests.
Your primary task is to first determine if the user's request is actually asking for a chart or visualization.

If the user's request does not explicitly ask for a chart or visualization, or if the request is unclear,
you must return ONLY this error response:
{{
    "is_error": true,
    "message": "I can only help with creating charts. Please ask me to create a specific type of chart (bar, line, pie, scatter) with the data you'd like to visualize."
}}

Your task is to:
1. Determine whether the user request is about creating a chart.
2. If the request is about creating a chart:
    2a. Understand what chart the user wants, including any filtering and customization requirements
    2b. Select appropriate dataset and columns
    2c. provide a JSON response with the specified schema. The response should be a valid JSON object matching the specified schema.
3. If the request is not related to creating a chart, or is unclear, return only:
{{
   "is_error": true,
    "message": "I can only help with creating charts. Please ask me to create a specific type of chart (bar, line, pie, scatter) with the data you'd like to visualize."
}}

Available datasets:
1. Sales data: Columns = Month, Revenue, Units, Category (Categories: Electronics, Clothing, Food)
2. Website traffic data: Columns = Day, Visitors, Bounce_Rate

For valid chart requests ONLY (where the user clearly asks for a visualization), provide a JSON response with all required field.
Optional fields include filter_colun, filter_value, and customization objects.

Example valid requests:
- "Show me a bar chart of monthly revenue"
- "Create a line graph of daily visitors"
- "Visualize the sales data"
- "Plot the Electronics revenue"

Example invalid requests:
- "What is cheese?"
- "Hello"
- "Tell me about sales"
- "test"
- "How are you?"

IMPORTANT: 
-Return an error if the request is not clearly about creating a chart.
-Only return valid JSON.
"""

    full_prompt = f"{context}\n\nUser request: {prompt}"

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type='application/json',
            temperature=0.1,
            response_schema=response_schema
        )
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Display raw response for debugging
        # st.write("Raw LLM Response:", response.text)

        json_data = json.loads(response.text)
        # st.write("json_data:", json_data)
        
        #vvalidate json output structure
        if "is_error" not in json_data or "message" not in json_data:
            return {"is_error": True, "message": "Invalid LLM response structure"}

         # Validate response structure for non-error responses
        if not json_data.get("is_error", True):          
            required_fields = ["chart_type", "dataset", "x_column", "y_column", "message"]
            if not all(field in json_data for field in required_fields):
                return {"is_error": True, "message": "Invalid response structure from LLM"}
                    
            valid_chart_types = ["bar", "line", "pie", "scatter"]
            if json_data["chart_type"] not in valid_chart_types:
                return {"is_error": True, "message": "Invalid chart type specified"}
                    
            valid_datasets = ["sales", "website_traffic"]
            if json_data["dataset"] not in valid_datasets:
                return {"is_error": True, "message": "Invalid dataset specified"}
        return json_data
        
    except json.JSONDecodeError as e:
        return {"is_error": True, "message": f"Failed to parse JSON response: {str(e)}"}
    except Exception as e:
        return {"is_error": True, "message": f"An error occurred while processing your request: {str(e)}"}

st.title("Chart Assistant")



st.markdown("""
Welcome! I'm an AI assistant for creating charts. I can help you visualize:
- Sales data (Revenue, Units by Month, filtered by Category: Electronics, Clothing, Food)
- Website traffic data (Visitors and Bounce Rate by Day)

Try asking for something like:
"Show me Electronics revenue trend over months in red color"
"Create a blue line chart of visitors with the title 'Daily Traffic Analysis'"
"Make a bar chart of monthly revenue with custom axis labels"
""")

# Chat input
if prompt := st.chat_input("What would you like to visualize?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from LLM
    llm_response = get_llm_response(prompt, get_sample_data())
    
    # Add assistant's response
    assistant_message = {
        "role": "assistant",
        "content": llm_response["message"],
        "is_error": llm_response["is_error"]
    }

    # If it's not an error, add chart configuration
    if not llm_response["is_error"]:
        assistant_message.update({
            "chart_config": {
                "chart_type": llm_response["chart_type"],
                "dataset": llm_response["dataset"],
                "x_column": llm_response["x_column"],
                "y_column": llm_response["y_column"],
                "filter_column": llm_response.get("filter_column"),
                "filter_value": llm_response.get("filter_value"),
                "customization": llm_response.get("customization")
            },
        })
    
    st.session_state.messages.append(assistant_message)

    # Add follow-up message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Would you like to create another chart? Feel free to ask!",
        "is_error": False
    })      

# Display chat history with inline charts
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Display the message content
        st.write(message["content"])
        
        st.write(f"Debug - Message {idx}:")
        st.write("- Raw message:", message)
        st.write("- has chart_config:","chart_config" in message)
        st.write("- is_error:", message.get("is_error", False))
        st.write("- role:", message.get("role", "unknown"))

        if (message["role"] == "assistant" and 
            not message["is_error"] and 
            "chart_config" in message):
            
            chart_id = st.session_state.chart_counter
            download_id = st.session_state.download_counter
            st.session_state.chart_counter += 1
            st.session_state.download_counter += 1

            fig = create_chart(
                message["chart_config"]["chart_type"],
                message["chart_config"]["dataset"],
                message["chart_config"]["x_column"],
                message["chart_config"]["y_column"],
                message["chart_config"]["filter_column"],
                message["chart_config"]["filter_value"],
                message["chart_config"]["customization"]
            )
            
            # Display the chart with unique key
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_id}")

            # Add download button with unique key
            png_img = get_chart_image(fig)
            st.download_button(
                label="Download PNG",
                data=png_img,
                file_name=f"chart_{chart_id}.png",
                mime="image/png",
                key=f"download_{download_id}"
            )