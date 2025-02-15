import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64
from datetime import datetime
import tiktoken
import json

# Page configuration
st.set_page_config(page_title="Multimodal Chat with Gemini", layout="wide", page_icon="🚀")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "image" not in st.session_state:
    st.session_state.image = None
if "model_config" not in st.session_state:
    st.session_state.model_config = {
        "model_name": "gemini-1.5-flash-latest",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful and friendly AI assistant developed by tanbaycu. "
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "font_size" not in st.session_state:
    st.session_state.font_size = "medium"

GEMINI_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-8b",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-lite-preview-02-05"
]

MAX_TOKENS = 8192  # Assuming this is the token limit for the model

# Custom CSS
def get_custom_css():
    return f"""
<style>
    .stButton > button {{width: 100%;}}
    .stTextInput > div > div > input {{background-color: {'#f0f2f6' if st.session_state.theme == 'light' else '#2b313e'};}}
    .sidebar .stButton > button {{background-color: #4CAF50; color: white;}}
    .sidebar .stButton > button:hover {{background-color: #45a049;}}
    .chat-message {{
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        font-size: {{'0.8rem' if st.session_state.font_size == 'small' else '1rem' if st.session_state.font_size == 'medium' else '1.2rem'}};
    }}
    .chat-message.user {{background-color: {'#e6f3ff' if st.session_state.theme == 'light' else '#2b313e'};}}
    .chat-message.bot {{background-color: {'#f0f0f0' if st.session_state.theme == 'light' else '#3c4354'};}}
    .chat-message .avatar {{width: 15%; padding-right: 0.5rem;}}
    .chat-message .avatar img {{max-width: 40px; max-height: 40px; border-radius: 50%;}}
    .chat-message .message {{width: 85%; padding: 0 1.5rem;}}
    .chat-message .timestamp {{font-size: 0.8em; color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'}; text-align: right; margin-top: 0.5rem;}}
    .token-info {{font-size: 0.8em; color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'}; margin-top: 0.5rem;}}
    body {{background-color: {'#ffffff' if st.session_state.theme == 'light' else '#1e1e1e'}; color: {'#000000' if st.session_state.theme == 'light' else '#ffffff'};}}
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Utility functions
@st.cache_resource
def load_model(api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name=model_name)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

@st.cache_data
def get_image_download_link(_img, filename, text):
    buffered = io.BytesIO()
    _img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def get_chat_history():
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-st.session_state.max_history:]])

@st.cache_data
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def handle_user_input(user_input, model):
    chat_history = get_chat_history()
    full_prompt = f"{st.session_state.system_prompt}\n\nChat history:\n{chat_history}\n\nUser: {user_input}\n\nAssistant:"
    
    inputs = [full_prompt]
    if st.session_state.image:
        inputs.append(st.session_state.image)
    
    try:
        response = model.generate_content(
            inputs,
            generation_config=genai.types.GenerationConfig(
                temperature=st.session_state.model_config["temperature"],
                top_p=st.session_state.model_config["top_p"],
                top_k=st.session_state.model_config["top_k"],
                max_output_tokens=st.session_state.model_config["max_output_tokens"],
            )
        )
        response_tokens = count_tokens(response.text)
        st.session_state.total_tokens += response_tokens
        return response.text, response_tokens
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None, 0

def save_chat_session():
    session_data = {
        "messages": st.session_state.messages,
        "chat_history": st.session_state.chat_history,
        "total_tokens": st.session_state.total_tokens
    }
    return json.dumps(session_data)

def load_chat_session(session_data):
    data = json.loads(session_data)
    st.session_state.messages = data["messages"]
    st.session_state.chat_history = data["chat_history"]
    st.session_state.total_tokens = data["total_tokens"]

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    api_key = st.text_input("Enter Google API Key", type="password")
    
    with st.expander("🛠️ Model Customization", expanded=False):
        selected_model = st.selectbox("Select Gemini model", GEMINI_MODELS, index=GEMINI_MODELS.index(st.session_state.model_config["model_name"]))
        st.session_state.model_config["model_name"] = selected_model
        
        st.session_state.model_config["temperature"] = st.slider("🌡️ Temperature", min_value=0.0, max_value=1.0, value=st.session_state.model_config["temperature"], step=0.1)
        st.session_state.model_config["top_p"] = st.slider("🎯 Top P", min_value=0.0, max_value=1.0, value=st.session_state.model_config["top_p"], step=0.1)
        st.session_state.model_config["top_k"] = st.number_input("🔝 Top K", min_value=1, max_value=100, value=st.session_state.model_config["top_k"])
        st.session_state.model_config["max_output_tokens"] = st.number_input("📏 Max Output Tokens", min_value=1, max_value=8192, value=st.session_state.model_config["max_output_tokens"])
    
    with st.expander("📝 Prompt Customization", expanded=False):
        st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=100)
    
    st.session_state.max_history = st.slider("🧠 Max messages in history", min_value=1, max_value=20, value=5)
    
    uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file)
        st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)
        st.markdown(get_image_download_link(st.session_state.image, "uploaded_image.png", "📥 Download Image"), unsafe_allow_html=True)

    # Hamburger menu for additional options
    with st.expander("☰ Advanced Options", expanded=False):
        st.subheader("Chat Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", key="clear_history"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.image = None
                st.session_state.total_tokens = 0
                st.rerun()

        with col2:
            if st.button("📥 Export", key="export_history"):
                chat_history = "\n".join([f"{msg['role']} ({msg.get('timestamp', 'N/A')}): {msg['content']}" for msg in st.session_state.chat_history])
                st.download_button(
                    label="📥 Download",
                    data=chat_history,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
        
        st.subheader("Save and Load Chat Session")
        if st.button("💾 Save"):
            session_data = save_chat_session()
            st.download_button(
                label="📥 Download Chat Session",
                data=session_data,
                file_name="chat_session.json",
                mime="application/json"
            )
        
        uploaded_session = st.file_uploader("📤 Upload", type=["json"])
        if uploaded_session is not None:
            session_data = uploaded_session.getvalue().decode("utf-8")
            load_chat_session(session_data)
            st.success("Chat session loaded successfully!")

        st.subheader("Model Information")
        st.info(f"""
        - 🤖 Model: {st.session_state.model_config['model_name']}
        - 🌡️ Temperature: {st.session_state.model_config['temperature']:.2f}
        - 🎯 Top P: {st.session_state.model_config['top_p']:.2f}
        - 🔝 Top K: {st.session_state.model_config['top_k']}
        - 📏 Max Output Tokens: {st.session_state.model_config['max_output_tokens']}
        - 🧠 Messages in history: {st.session_state.max_history}
        - 💬 Total messages: {len(st.session_state.messages)}
        - 🔢 Total tokens: {st.session_state.total_tokens}
        """)

        # Token usage progress bar
        st.subheader("Token Usage")
        progress = st.session_state.total_tokens / MAX_TOKENS
        st.progress(progress)
        st.text(f"{st.session_state.total_tokens}/{MAX_TOKENS} tokens used")

# Main content
st.title("🚀 Gemini Agent")
st.caption("Experience the power of the latest Gemini models with advanced customization. 🌟")

if api_key:
    model = load_model(api_key, st.session_state.model_config["model_name"])
    
    if model:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                timestamp = msg.get('timestamp', 'N/A')
                tokens = msg.get('tokens', 'N/A')
                st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {tokens}</div>", unsafe_allow_html=True)

        # Handle user input
        prompt = st.chat_input("💬 What would you like to know?")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
                tokens = count_tokens(prompt)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {tokens}</div>", unsafe_allow_html=True)
                st.session_state.total_tokens += tokens
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt,
                    "tokens": tokens,
                    "timestamp": timestamp
                })

            with st.chat_message("assistant"):
                with st.spinner('🤔 Generating response...'):
                    response, response_tokens = handle_user_input(prompt, model)
                    if response:
                        st.markdown(response)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {response_tokens}</div>", unsafe_allow_html=True)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "tokens": response_tokens,
                            "timestamp": timestamp
                        })

        # Warning if there's an image but no prompt
        if st.session_state.image and not prompt:
            st.warning("⚠️ Please enter a question to go along with the image.")
    else:
        st.error("❌ Unable to initialize the model. Please check your API key and try again.")
else:
    st.warning("🔑 Please enter your Google API Key in the sidebar to start chatting.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by tanbaycu")

