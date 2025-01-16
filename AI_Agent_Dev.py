import streamlit as st
import asyncio
import aiohttp
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app_logs.log"), logging.StreamHandler()],
)

# Configure Streamlit page and sidebar
st.set_page_config(page_title="AI-Powered Code Builder", layout="wide", initial_sidebar_state="collapsed")

# Get API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]

# Update the sidebar API key inputs to use secrets
st.sidebar.header("API Configuration")
OPENAI_API_KEY = st.sidebar.text_input(
    "OpenAI API Key", 
    value=OPENAI_API_KEY,
    type="password",
    key="openai_api_key_input"
)
GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key", 
    value=GEMINI_API_KEY,
    type="password",
    key="gemini_api_key_input"
)
CLAUDE_API_KEY = st.sidebar.text_input(
    "Claude API Key", 
    value=CLAUDE_API_KEY,
    type="password",
    key="claude_api_key_input"
)

# Streamlit App Configuration
st.title("AI-Powered Code Builder")

# Initialize Session State (move this section before any UI elements)
if "selected_models" not in st.session_state:
    st.session_state.selected_models = {
        "ideation": "gemini",  # Default to current setup
        "feedback": "gemini",
        "implementation": "claude",
        "enhancement": "openai"
    }

if "steps" not in st.session_state:
    st.session_state.steps = {
        "idea": None,
        "feedback": None,
        "code": None,
        "enhanced_code": None,
    }

if "progress" not in st.session_state:
    st.session_state.progress = 0

if "expander_states" not in st.session_state:
    st.session_state.expander_states = {
        "step1": True,
        "step2": False,
        "step3": False,
        "step4": False,
        "step5": False,
        "step6": False
    }

if "custom_prompts" not in st.session_state:
    st.session_state.custom_prompts = {
        "ideation": "",
        "feedback": "",
        "implementation": "",
        "enhancement": ""
    }

# Move this initialization before any UI elements that use it
if "expander_titles" not in st.session_state:
    st.session_state.expander_titles = {
        "step1": f"Step 1: App Ideation ({st.session_state.selected_models['ideation'].upper()}) 🎯",
        "step2": f"Step 2: Product Manager Review ({st.session_state.selected_models['feedback'].upper()}) 📊",
        "step3": f"Step 3: Implementation ({st.session_state.selected_models['implementation'].upper()}) 💻",
        "step4": f"Step 4: Code Enhancement ({st.session_state.selected_models['enhancement'].upper()}) ⚡"
    }

# Progress Bar
progress_bar = st.progress(st.session_state.progress)

# API Configuration
import openai

# Add to the sidebar customization section
with st.sidebar.expander("AI Model Selection"):
    st.write("**Select AI Model for Each Step**")
    
    # Ensure selected_models exists in session state
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = {
            "ideation": "gemini",
            "feedback": "gemini",
            "implementation": "claude",
            "enhancement": "openai"
        }
    
    # Model selection dropdowns with safer access
    current_ideation = st.session_state.selected_models.get("ideation", "gemini")
    current_feedback = st.session_state.selected_models.get("feedback", "gemini")
    current_implementation = st.session_state.selected_models.get("implementation", "claude")
    current_enhancement = st.session_state.selected_models.get("enhancement", "openai")
    
    # Update model selections
    st.session_state.selected_models["ideation"] = st.selectbox(
        "Step 1: App Ideation Model",
        options=["gemini", "openai", "claude"],
        index=["gemini", "openai", "claude"].index(current_ideation),
        key="ideation_model"
    )
    
    st.session_state.selected_models["feedback"] = st.selectbox(
        "Step 2: Feedback Model",
        options=["gemini", "openai", "claude"],
        index=["gemini", "openai", "claude"].index(current_feedback),
        key="feedback_model"
    )
    
    st.session_state.selected_models["implementation"] = st.selectbox(
        "Step 3: Implementation Model",
        options=["claude", "openai", "gemini"],
        index=["claude", "openai", "gemini"].index(current_implementation),
        key="implementation_model"
    )
    
    st.session_state.selected_models["enhancement"] = st.selectbox(
        "Step 4: Enhancement Model",
        options=["openai", "claude", "gemini"],
        index=["openai", "claude", "gemini"].index(current_enhancement),
        key="enhancement_model"
    )

    # Update the Apply button to properly update session state
    if st.button("Apply Model Changes", key="apply_models"):
        # Update the expander titles
        st.session_state.expander_titles = {
            "step1": f"Step 1: App Ideation ({st.session_state.selected_models['ideation'].upper()}) 🎯",
            "step2": f"Step 2: Product Manager Review ({st.session_state.selected_models['feedback'].upper()}) 📊",
            "step3": f"Step 3: Implementation ({st.session_state.selected_models['implementation'].upper()}) 💻",
            "step4": f"Step 4: Code Enhancement ({st.session_state.selected_models['enhancement'].upper()}) ⚡"
        }
        st.rerun()

# Add new helper functions for model-agnostic API calls
async def call_openai_api(prompt, model="gpt-4", system_message="You are a helpful assistant."):
    """Generic function for OpenAI API calls."""
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        st.error(f"OpenAI API error: {str(e)}")
        return None

async def call_gemini_api(prompt):
    """Generic function for Gemini API calls."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            ) as response:
                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        st.error(f"Gemini API error: {str(e)}")
        return None

async def call_claude_api(prompt):
    """Generic function for Claude API calls."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048
                }
            ) as response:
                result = await response.json()
                return result["content"][0]["text"]
    except Exception as e:
        logging.error(f"Claude API error: {str(e)}")
        st.error(f"Claude API error: {str(e)}")
        return None

# Rename the function to be model-agnostic
async def generate_app_idea(prompt):
    """Generates app ideas using selected AI model."""
    full_prompt = f"{st.session_state.active_prompts['ideation']}\n\nThe app should be related to: {prompt}"
    
    model = st.session_state.selected_models["ideation"]
    if model == "gemini":
        return await call_gemini_api(full_prompt)
    elif model == "openai":
        return await call_openai_api(full_prompt)
    else:  # claude
        return await call_claude_api(full_prompt)

async def get_feedback_on_idea(idea):
    """Gets feedback using selected AI model."""
    full_prompt = f"{st.session_state.active_prompts['feedback']}\n\nApp Idea: {idea}"
    
    model = st.session_state.selected_models["feedback"]
    if model == "gemini":
        return await call_gemini_api(full_prompt)
    elif model == "openai":
        return await call_openai_api(full_prompt)
    else:  # claude
        return await call_claude_api(full_prompt)

async def implement_code(idea, feedback):
    """Implements code using selected AI model."""
    full_prompt = f"{st.session_state.active_prompts['implementation']}\n\nIdea: {idea}\n\nFeedback to consider: {feedback}"
    
    model = st.session_state.selected_models["implementation"]
    if model == "gemini":
        return await call_gemini_api(full_prompt)
    elif model == "openai":
        return await call_openai_api(full_prompt)
    else:  # claude
        return await call_claude_api(full_prompt)

async def review_and_enhance_code(code):
    """Enhances code using selected AI model."""
    full_prompt = f"{st.session_state.active_prompts['enhancement']}\n\n{code}"
    
    model = st.session_state.selected_models["enhancement"]
    if model == "gemini":
        return await call_gemini_api(full_prompt)
    elif model == "openai":
        return await call_openai_api(full_prompt)
    else:  # claude
        return await call_claude_api(full_prompt)

# Update the process_workflow function
async def process_workflow(prompt):
    """Main workflow to process user input and call AI models."""
    try:
        st.session_state.progress = 0
        progress_bar.progress(st.session_state.progress)
        
        result = None
        total_steps = len(st.session_state.workflow_steps)
        progress_increment = 1.0 / total_steps  # Changed to ensure value between 0 and 1

        for step in st.session_state.workflow_steps:
            with st.expander(f"{step['title']} ({step['model'].upper()})", 
                           expanded=True):
                st.write(f"*Processing using {step['model'].upper()}...*")
                
                # Prepare the prompt
                step_prompt = step['prompt'] or step['default_prompt']
                if result:  # Add previous result to prompt if it exists
                    full_prompt = f"{step_prompt}\n\nPrevious result: {result}\n\nUser input: {prompt}"
                else:
                    full_prompt = f"{step_prompt}\n\nUser input: {prompt}"

                # Call appropriate API based on selected model
                if step['model'] == "gemini":
                    result = await call_gemini_api(full_prompt)
                elif step['model'] == "openai":
                    result = await call_openai_api(full_prompt)
                else:  # claude
                    result = await call_claude_api(full_prompt)

                if not result:
                    st.error(f"Failed at {step['title']}. Please try again.")
                    return None

                st.markdown(f"**Output:**\n{result}")
                
                # Update progress (ensure it stays between 0 and 1)
                st.session_state.progress = min(st.session_state.progress + progress_increment, 1.0)
                progress_bar.progress(st.session_state.progress)

        return result

    except Exception as e:
        logging.error(f"Error in workflow: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return None

# Add after the CSS styling and before using the function
def get_model_html(model_name):
    return f'<div class="model-{model_name.lower()}">Using: {model_name.upper()}</div>'

def get_step_html(step_number, title, model_name, description):
    return f"""
    <div class="step-container">
        <h3>Step {step_number} {title}</h3>
        <strong>{description}</strong><br><br>
        {get_model_html(model_name)}
    </div>
    """

# Make sure this CSS is defined before the functions
st.markdown("""
<style>
    .model-gemini {
        background-color: #e6f3ff;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d9ff;
    }
    .model-openai {
        background-color: #e6ffe6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #b3ffb3;
    }
    .model-claude {
        background-color: #fff2e6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #ffd9b3;
    }
    .step-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        height: 100%;
    }
    .card-title {
        min-height: 50px;  /* Fixed height for title area */
        margin-bottom: 10px;
    }
    
    .card-description {
        min-height: 60px;  /* Fixed height for description area */
    }
    
    .workflow-card {
        border: 2px solid;
        padding: 15px;
        border-radius: 8px;
        height: 100%;
    }
    
    .model-indicator {
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        background-color: #f8f9fa;
    }
    
    /* Model-specific border colors */
    .workflow-card.model-gemini { border-color: #1a73e8; }
    .workflow-card.model-openai { border-color: #10a37f; }
    .workflow-card.model-claude { border-color: #7c3aed; }
    
    hr.model-gemini { border-color: #1a73e8; }
    hr.model-openai { border-color: #10a37f; }
    hr.model-claude { border-color: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# Update the workflow preview section to control the actual workflow
st.write("## Workflow Preview")
st.write("Here's how your app will be built:")

# Define default prompts at the beginning of the file
default_ideation = """You are an expert Streamlit app creator with a proven track record of developing successful, 
highly monetizable Python-based web applications. Generate ONE innovative Streamlit app idea that:
1. Leverages Streamlit's strengths for rapid deployment and interactive data applications
2. Can be implemented efficiently in Python
3. Has clear monetization potential
4. Provides unique value through AI/ML capabilities
5. Can scale effectively using Streamlit's cloud infrastructure"""

default_feedback = """Review this app idea and provide BRIEF, SPECIFIC feedback in 3-4 bullet points. 
Focus only on critical improvements or concerns. Be specific and concise."""

default_implementation = """Please write Python code to implement the following app idea.
Please provide only the implementation code with clear comments.
Focus on creating a working prototype that demonstrates the core functionality."""

default_enhancement = """Review and enhance the following Python code. Focus on:
1. Code optimization and efficiency
2. Best practices and patterns
3. Error handling and robustness
4. Documentation and clarity"""

# Initialize workflow steps after defining default prompts
if "workflow_steps" not in st.session_state:
    st.session_state.workflow_steps = [
        {
            "id": 1,
            "title": "🎯 App Ideation",
            "description": "Generates innovative app concept based on your description",
            "model": "gemini",
            "prompt": "",
            "default_prompt": default_ideation
        },
        {
            "id": 2,
            "title": "📊 Product Review",
            "description": "Analyzes feasibility and provides strategic feedback",
            "model": "gemini",
            "prompt": "",
            "default_prompt": default_feedback
        },
        {
            "id": 3,
            "title": "💻 Implementation",
            "description": "Generates working code implementation",
            "model": "claude",
            "prompt": "",
            "default_prompt": default_implementation
        },
        {
            "id": 4,
            "title": "⚡ Enhancement",
            "description": "Optimizes and improves the code",
            "model": "openai",
            "prompt": "",
            "default_prompt": default_enhancement
        }
    ]

# Create dynamic columns based on number of steps
num_steps = len(st.session_state.workflow_steps)
cols = st.columns(num_steps)

for idx, step in enumerate(st.session_state.workflow_steps):
    with cols[idx]:
        # Step container with remove button
        st.markdown(
            f"""
            <div class="workflow-card model-{step['model']}">
                <div class="card-header">
                    <div class="card-title">{step['title']}</div>
                    <hr class="model-{step['model']}" style="margin: 8px 0; border-width: 1px;">
                    <div class="card-description">{step['description']}</div>
                    <div class="model-indicator">Using: {step['model'].upper()}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Model selection for this step
        step["model"] = st.selectbox(
            "Select Model",
            options=["gemini", "openai", "claude"],
            index=["gemini", "openai", "claude"].index(step["model"]),
            key=f"model_select_{step['id']}"
        )
        
        # Custom prompt for this step
        step["prompt"] = st.text_area(
            "Custom Prompt",
            value=step["prompt"],
            placeholder=step["default_prompt"],
            help="Leave blank to use default prompt",
            key=f"prompt_{step['id']}",
            height=150
        )
        
        # Remove button for this step
        if st.button("❌ Remove Step", key=f"remove_{step['id']}", use_container_width=True):
            if len(st.session_state.workflow_steps) > 1:  # Prevent removing all steps
                st.session_state.workflow_steps = [s for s in st.session_state.workflow_steps if s['id'] != step['id']]
                st.rerun()

# Add "Add Step" button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("➕ Add New Step", use_container_width=True):
        new_id = len(st.session_state.workflow_steps) + 1
        new_step = {
            "id": new_id,
            "title": f"🔧 Custom Step {new_id}",
            "description": "Custom processing step",
            "model": "gemini",
            "prompt": "",
            "default_prompt": "Enter your custom prompt here..."
        }
        st.session_state.workflow_steps.append(new_step)
        st.rerun()

# Add apply button for all changes
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Apply Changes", key="apply_changes", use_container_width=True):
        # Update session state with current values
        st.session_state.selected_models = {
            f"step_{step['id']}": step["model"] for step in st.session_state.workflow_steps
        }
        st.session_state.custom_prompts = {
            f"step_{step['id']}": step["prompt"] for step in st.session_state.workflow_steps
        }
        st.success("Changes applied successfully!")
        st.rerun()

st.divider()

# Add a divider before the user input
st.write("## Generate Your App")
user_input = st.text_input("Enter a brief description of the app you want to build:")

# Add this before the user input section
st.sidebar.header("Customize AI Prompts")
with st.sidebar.expander("Default Prompts & Customization"):
    st.write("**Customize AI Prompts**")
    
    # Ensure custom_prompts exists in session state
    if "custom_prompts" not in st.session_state:
        st.session_state.custom_prompts = {
            "ideation": "",
            "feedback": "",
            "implementation": "",
            "enhancement": ""
        }
    
    # Ideation prompt
    st.write("**Step 1: App Ideation Prompt**")
    default_ideation = """You are an expert Streamlit app creator with a proven track record of developing successful, 
    highly monetizable Python-based web applications. Generate ONE innovative Streamlit app idea that:
    1. Leverages Streamlit's strengths for rapid deployment and interactive data applications
    2. Can be implemented efficiently in Python
    3. Has clear monetization potential
    4. Provides unique value through AI/ML capabilities
    5. Can scale effectively using Streamlit's cloud infrastructure"""
    st.session_state.custom_prompts["ideation"] = st.text_area(
        "Custom Ideation Prompt", 
        value=st.session_state.custom_prompts.get("ideation", ""),
        placeholder=default_ideation,
        help="Leave blank to use default prompt",
        key="ideation_prompt"
    )

    # Feedback prompt
    st.write("**Step 2: Feedback Prompt**")
    default_feedback = """Review this app idea and provide BRIEF, SPECIFIC feedback in 3-4 bullet points. 
    Focus only on critical improvements or concerns. Be specific and concise."""
    st.session_state.custom_prompts["feedback"] = st.text_area(
        "Custom Feedback Prompt",
        value=st.session_state.custom_prompts.get("feedback", ""),
        placeholder=default_feedback,
        help="Leave blank to use default prompt",
        key="feedback_prompt"
    )

    # Implementation prompt
    st.write("**Step 3: Implementation Prompt**")
    default_implementation = """Please write Python code to implement the following app idea.
    Please provide only the implementation code with clear comments.
    Focus on creating a working prototype that demonstrates the core functionality."""
    st.session_state.custom_prompts["implementation"] = st.text_area(
        "Custom Implementation Prompt",
        value=st.session_state.custom_prompts.get("implementation", ""),
        placeholder=default_implementation,
        help="Leave blank to use default prompt",
        key="implementation_prompt"
    )

    # Enhancement prompt
    st.write("**Step 4: Enhancement Prompt**")
    default_enhancement = """Review and enhance the following Python code. Focus on:
    1. Code optimization and efficiency
    2. Best practices and patterns
    3. Error handling and robustness
    4. Documentation and clarity"""
    st.session_state.custom_prompts["enhancement"] = st.text_area(
        "Custom Enhancement Prompt",
        value=st.session_state.custom_prompts.get("enhancement", ""),
        placeholder=default_enhancement,
        help="Leave blank to use default prompt",
        key="enhancement_prompt"
    )

    # Add apply button for prompts
    if st.button("Apply Prompt Changes", key="apply_prompts"):
        # Store the new prompts
        st.session_state.active_prompts = {
            "ideation": st.session_state.custom_prompts.get("ideation", "") or default_ideation,
            "feedback": st.session_state.custom_prompts.get("feedback", "") or default_feedback,
            "implementation": st.session_state.custom_prompts.get("implementation", "") or default_implementation,
            "enhancement": st.session_state.custom_prompts.get("enhancement", "") or default_enhancement
        }
        
        st.success("Prompt changes applied successfully!")
        st.rerun()

# Initialize active prompts if not exists
if "active_prompts" not in st.session_state:
    st.session_state.active_prompts = {
        "ideation": default_ideation,
        "feedback": default_feedback,
        "implementation": default_implementation,
        "enhancement": default_enhancement
    }

if st.button("Generate Code"):
    if user_input:
        enhanced_code = asyncio.run(process_workflow(user_input))

        # Download Option
        if enhanced_code:
            st.download_button(
                label="Download Enhanced Code",
                data=enhanced_code,
                file_name="app_code.py",
                mime="text/x-python",
            )
    else:
        st.warning("Please enter a description of your app.")

# Display Logs
with st.expander("Show Logs"):
    with open("app_logs.log", "r") as log_file:
        st.text(log_file.read())
