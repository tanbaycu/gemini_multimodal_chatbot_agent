import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64
from datetime import datetime
import tiktoken
import json
import time
import re
import os
import logging
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt

# Cấu hình logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình trang
st.set_page_config(page_title="Trò chuyện Đa phương tiện với Gemini", layout="wide", page_icon="🚀")

# Khởi tạo trạng thái phiên
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
    st.session_state.system_prompt = "Bạn là một trợ lý AI hữu ích và thân thiện được phát triển bởi tanbaycu. "
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "font_size" not in st.session_state:
    st.session_state.font_size = "medium"
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Mặc định"

GEMINI_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-8b",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-lite-preview-02-05"
]

MAX_TOKENS = 8192  # Giả sử đây là giới hạn token cho mô hình

# Cải thiện giao diện người dùng
def get_custom_css():
    return f"""
<style>
    .stButton > button {{
        width: 100%;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .stTextInput > div > div > input {{
        background-color: {'#f0f2f6' if st.session_state.theme == 'light' else '#2b313e'};
    }}
    .sidebar .stButton > button {{
        background-color: #4CAF50;
        color: white;
    }}
    .sidebar .stButton > button:hover {{
        background-color: #45a049;
    }}
    .chat-message {{
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        font-size: {{'0.8rem' if st.session_state.font_size == 'small' else '1rem' if st.session_state.font_size == 'medium' else '1.2rem'}};
        animation: fadeIn 0.5s;
    }}
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    .chat-message.user {{
        background-color: {'#e6f3ff' if st.session_state.theme == 'light' else '#2b313e'};
    }}
    .chat-message.bot {{
        background-color: {'#f0f0f0' if st.session_state.theme == 'light' else '#3c4354'};
    }}
    .chat-message .avatar {{
        width: 15%;
        padding-right: 0.5rem;
    }}
    .chat-message .avatar img {{
        max-width: 40px;
        max-height: 40px;
        border-radius: 50%;
    }}
    .chat-message .message {{
        width: 85%;
        padding: 0 1.5rem;
    }}
    .chat-message .timestamp {{
        font-size: 0.8em;
        color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'};
        text-align: right;
        margin-top: 0.5rem;
    }}
    .token-info {{
        font-size: 0.8em;
        color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'};
        margin-top: 0.5rem;
    }}
    body {{
        transition: background-color 0.3s ease, color 0.3s ease;
        background-color: {'#ffffff' if st.session_state.theme == 'light' else '#1e1e1e'};
        color: {'#000000' if st.session_state.theme == 'light' else '#ffffff'};
    }}
    .stAlert {{
        animation: slideIn 0.5s;
    }}
    @keyframes slideIn {{
        0% {{ transform: translateY(-100%); }}
        100% {{ transform: translateY(0); }}
    }}
    
    /* Cải thiện khả năng phản hồi trên các thiết bị di động */
    @media (max-width: 768px) {{
        .chat-message {{
            flex-direction: column;
        }}
        .chat-message .avatar {{
            width: 100%;
            margin-bottom: 0.5rem;
        }}
        .chat-message .message {{
            width: 100%;
        }}
    }}
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Tối ưu hóa hiệu suất
@st.cache_resource
def load_model(api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name=model_name)
    except Exception as e:
        logging.error(f"Lỗi khởi tạo mô hình: {str(e)}")
        st.error(f"Lỗi khởi tạo mô hình: {str(e)}")
        return None

@st.cache_data
def get_image_download_link(_img, filename, text):
    buffered = io.BytesIO()
    _img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

@st.cache_data
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Cải thiện cơ chế giới hạn tốc độ
def rate_limited_response(func):
    def wrapper(*args, **kwargs):
        if 'last_request_time' not in st.session_state:
            st.session_state.last_request_time = 0
        
        current_time = time.time()
        time_since_last_request = current_time - st.session_state.last_request_time
        
        if time_since_last_request < 0.5:  # Giới hạn 2 yêu cầu mỗi giây
            time.sleep(0.5 - time_since_last_request)
        
        result = func(*args, **kwargs)
        st.session_state.last_request_time = time.time()
        return result
    return wrapper

# Quản lý phiên
def save_session(session_name):
    st.session_state.sessions[session_name] = {
        "messages": st.session_state.messages,
        "chat_history": st.session_state.chat_history,
        "total_tokens": st.session_state.total_tokens,
        "model_config": st.session_state.model_config,
        "system_prompt": st.session_state.system_prompt
    }
    logging.info(f"Đã lưu phiên '{session_name}'")
    st.success(f"Đã lưu phiên '{session_name}'")

def load_session(session_name):
    if session_name in st.session_state.sessions:
        session_data = st.session_state.sessions[session_name]
        st.session_state.messages = session_data["messages"]
        st.session_state.chat_history = session_data["chat_history"]
        st.session_state.total_tokens = session_data["total_tokens"]
        st.session_state.model_config = session_data["model_config"]
        st.session_state.system_prompt = session_data["system_prompt"]
        st.session_state.current_session = session_name
        logging.info(f"Đã tải phiên '{session_name}'")
        st.success(f"Đã tải phiên '{session_name}'")
    else:
        logging.warning(f"Không tìm thấy phiên '{session_name}'")
        st.error(f"Không tìm thấy phiên '{session_name}'")

def search_chat_history(query):
    results = []
    for msg in st.session_state.chat_history:
        if query.lower() in msg['content'].lower():
            results.append(msg)
    return results

# Xử lý lỗi và phản hồi
def handle_error(error_message):
    logging.error(error_message)
    st.error(f"Lỗi: {error_message}")
    st.info("Vui lòng thử lại hoặc liên hệ hỗ trợ nếu lỗi vẫn tiếp tục.")

# Hàm xử lý input của người dùng (đã được cải tiến)
@rate_limited_response
def handle_user_input(user_input, model):
    start_time = time.time()
    sanitized_input = sanitize_input(user_input)
    chat_history = get_chat_history()
    full_prompt = f"{st.session_state.system_prompt}\n\nLịch sử trò chuyện:\n{chat_history}\n\nNgười dùng: {sanitized_input}\n\nTrợ lý:"
    
    inputs = [full_prompt]
    if st.session_state.image:
        inputs.append(st.session_state.image)
    
    try:
        with st.spinner('🤔 Đang tạo phản hồi...'):
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
        processing_time = time.time() - start_time
        logging.info(f"Phản hồi được tạo trong {processing_time:.2f} giây, sử dụng {response_tokens} tokens")
        return response.text, response_tokens, processing_time
    except Exception as e:
        error_message = f"Lỗi tạo phản hồi: {str(e)}"
        logging.error(error_message)
        handle_error(error_message)
        return None, 0, 0

def sanitize_input(text):
    # Loại bỏ các thẻ HTML
    text = re.sub('<[^<]+?>', '', text)
    # Escape các ký tự đặc biệt
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    return text

def get_chat_history():
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-st.session_state.max_history:]])

# Thanh bên (với các cải tiến)
with st.sidebar:
    st.title("Cài đặt")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    api_key = st.text_input("Nhập Google API Key", type="password", value=st.session_state.api_key)
    
    # Chế độ tối/sáng
    theme = st.radio("Chọn chủ đề", ["Sáng", "Tối"])
    st.session_state.theme = "light" if theme == "Sáng" else "dark"
    
    with st.expander("🛠️ Tùy chỉnh Mô hình", expanded=False):
        selected_model = st.selectbox("Chọn mô hình Gemini", GEMINI_MODELS, index=GEMINI_MODELS.index(st.session_state.model_config["model_name"]))
        st.session_state.model_config["model_name"] = selected_model
        
        st.session_state.model_config["temperature"] = st.slider("🌡️ Độ sáng tạo", min_value=0.0, max_value=1.0, value=st.session_state.model_config["temperature"], step=0.1)
        st.session_state.model_config["top_p"] = st.slider("🎯 Top P", min_value=0.0, max_value=1.0, value=st.session_state.model_config["top_p"], step=0.1)
        st.session_state.model_config["top_k"] = st.number_input("🔝 Top K", min_value=1, max_value=100, value=st.session_state.model_config["top_k"])
        st.session_state.model_config["max_output_tokens"] = st.number_input("📏 Số token tối đa", min_value=1, max_value=8192, value=st.session_state.model_config["max_output_tokens"])
    
    with st.expander("📝 Tùy chỉnh Prompt", expanded=False):
        st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=100)
    
    st.session_state.max_history = st.slider("🧠 Số lượng tin nhắn tối đa trong lịch sử", min_value=1, max_value=50, value=5)
    
    uploaded_file = st.file_uploader("📸 Tải lên một hình ảnh...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file)
        st.image(st.session_state.image, caption='Hình ảnh đã tải lên', use_column_width=True)
        st.markdown(get_image_download_link(st.session_state.image, "hình_ảnh_đã_tải.png", "📥 Tải xuống hình ảnh"), unsafe_allow_html=True)

    # Quản lý phiên
    with st.expander("💾 Quản lý phiên", expanded=False):
        session_name = st.text_input("Tên phiên")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Lưu phiên"):
                if session_name:
                    save_session(session_name)
                else:
                    st.warning("Vui lòng nhập tên phiên")
        with col2:
            if st.button("Tải phiên"):
                if session_name:
                    load_session(session_name)
                else:
                    st.warning("Vui lòng nhập tên phiên")
        
        if st.session_state.sessions:
            selected_session = st.selectbox("Chọn phiên để tải", list(st.session_state.sessions.keys()))
            if st.button("Tải phiên đã chọn"):
                load_session(selected_session)

        # Tìm kiếm trong lịch sử trò chuyện
        search_query = st.text_input("Tìm kiếm trong lịch sử trò chuyện")
        if search_query:
            search_results = search_chat_history(search_query)
            if search_results:
                st.write("Kết quả tìm kiếm:")
                for result in search_results:
                    st.write(f"{result['role']}: {result['content'][:100]}...")
            else:
                st.write("Không tìm thấy kết quả.")

# Nội dung chính (với các cải tiến)
st.title("🚀 Gemini Agent")
st.caption("Trải nghiệm sức mạnh của các mô hình Gemini mới nhất với tùy chỉnh nâng cao. 🌟")

st.write(f"Phiên hiện tại: {st.session_state.current_session}")

if api_key:
    if len(api_key) > 10:  # Kiểm tra đơn giản về tính hợp lệ của API key
        st.session_state.api_key = api_key
        model = load_model(api_key, st.session_state.model_config["model_name"])
        
        if model:
            # Hiển thị lịch sử trò chuyện
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    timestamp = msg.get('timestamp', 'N/A')
                    tokens = msg.get('tokens', 'N/A')
                    processing_time = msg.get('processing_time', None)
                    
                    info_text = f"{timestamp} | Tokens: {tokens}"
                    if processing_time and msg["role"] == "assistant":
                        formatted_time = f"{processing_time:.1f}s"
                        info_text += f" | {formatted_time}"
                    
                    st.markdown(f"<div class='timestamp'>{info_text}</div>", unsafe_allow_html=True)

            # Xử lý input của người dùng
            prompt = st.chat_input("💬 Bạn muốn biết gì?")
            if prompt:
                user_timestamp = datetime.now().strftime("%H:%M:%S")
                user_tokens = count_tokens(prompt)
                
                with st.chat_message("user"):
                    st.markdown(sanitize_input(prompt))
                    st.markdown(f"<div class='timestamp'>{user_timestamp} | Tokens: {user_tokens}</div>", unsafe_allow_html=True)
                
                st.session_state.total_tokens += user_tokens
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": sanitize_input(prompt),
                    "tokens": user_tokens,
                    "timestamp": user_timestamp
                })

                with st.chat_message("assistant"):
                    response, response_tokens, processing_time = handle_user_input(prompt, model)
                    if response:
                        st.markdown(response)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_time = f"{processing_time:.1f}s"
                        st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {response_tokens} | {formatted_time}</div>", unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "tokens": response_tokens,
                            "timestamp": timestamp,
                            "processing_time": processing_time
                        })

            # Cảnh báo nếu có hình ảnh nhưng không có prompt
            if st.session_state.image and not prompt:
                st.warning("⚠️ Vui lòng nhập câu hỏi để đi kèm với hình ảnh.")
        else:
            handle_error("Không thể khởi tạo mô hình. Vui lòng kiểm tra API key và thử lại.")
    else:
        handle_error("API key không hợp lệ. Vui lòng nhập một Google API Key hợp lệ.")
else:
    st.warning("🔑 Vui lòng nhập Google API Key của bạn ở thanh bên để bắt đầu trò chuyện.")

# Hiển thị biểu đồ thống kê
if st.button("📊 Hiển thị thống kê"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Biểu đồ phân bố tin nhắn
    roles = [msg['role'] for msg in st.session_state.chat_history]
    role_counts = {role: roles.count(role) for role in set(roles)}
    ax1.pie(role_counts.values(), labels=role_counts.keys(), autopct='%1.1f%%')
    ax1.set_title('Phân bố tin nhắn')
    
    # Biểu đồ sử dụng token
    tokens = [msg.get('tokens', 0) for msg in st.session_state.chat_history]
    ax2.plot(range(len(tokens)), tokens)
    ax2.set_title('Sử dụng token theo thời gian')
    ax2.set_xlabel('Số thứ tự tin nhắn')
    ax2.set_ylabel('Số token')
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Được phát triển với ❤️ bởi tanbaycu")