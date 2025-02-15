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
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

GEMINI_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-8b",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-lite-preview-02-05"
]

MAX_TOKENS = 8192

# CSS tùy chỉnh
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
        animation: fadeIn 0.5s;
    }}
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    .chat-message.user {{background-color: {'#e6f3ff' if st.session_state.theme == 'light' else '#2b313e'};}}
    .chat-message.bot {{background-color: {'#f0f0f0' if st.session_state.theme == 'light' else '#3c4354'};}}
    .chat-message .avatar {{width: 15%; padding-right: 0.5rem;}}
    .chat-message .avatar img {{max-width: 40px; max-height: 40px; border-radius: 50%;}}
    .chat-message .message {{width: 85%; padding: 0 1.5rem;}}
    .chat-message .timestamp {{font-size: 0.8em; color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'}; text-align: right; margin-top: 0.5rem;}}
    .token-info {{font-size: 0.8em; color: {'#a0a0a0' if st.session_state.theme == 'light' else '#cccccc'}; margin-top: 0.5rem;}}
    body {{background-color: {'#ffffff' if st.session_state.theme == 'light' else '#1e1e1e'}; color: {'#000000' if st.session_state.theme == 'light' else '#ffffff'};}}
    .stAlert {{animation: slideIn 0.5s;}}
    @keyframes slideIn {{
        0% {{ transform: translateY(-100%); }}
        100% {{ transform: translateY(0); }}
    }}
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Các hàm tiện ích
@st.cache_resource
def load_model(api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name=model_name)
    except Exception as e:
        st.error(f"Lỗi khởi tạo mô hình: {str(e)}")
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

def rate_limited_response(func):
    def wrapper(*args, **kwargs):
        if 'last_request_time' not in st.session_state:
            st.session_state.last_request_time = 0
        
        current_time = time.time()
        time_since_last_request = current_time - st.session_state.last_request_time
        
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        
        result = func(*args, **kwargs)
        st.session_state.last_request_time = time.time()
        return result
    return wrapper

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
        return response.text, response_tokens, processing_time
    except Exception as e:
        st.error(f"Lỗi tạo phản hồi: {str(e)}")
        return None, 0, 0

def sanitize_input(text):
    text = re.sub('<[^<]+?>', '', text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
    return text

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

def is_valid_api_key(api_key):
    return bool(api_key) and len(api_key) > 10

def format_processing_time(seconds):
    return f"{seconds:.1f}s"

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Kiểm tra xem bảng users đã tồn tại chưa
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = c.fetchone()
    
    if table_exists:
        # Kiểm tra số cột hiện tại
        c.execute("PRAGMA table_info(users)")
        columns = c.fetchall()
        if len(columns) == 3:  # Nếu chỉ có 3 cột
            # Thêm cột email
            c.execute("ALTER TABLE users ADD COLUMN email TEXT")
            conn.commit()
            print("Đã thêm cột email vào bảng users")
    else:
        # Tạo bảng mới với 4 cột
        c.execute('''CREATE TABLE users
                     (username TEXT PRIMARY KEY, password TEXT, api_key TEXT, email TEXT)''')
        conn.commit()
        print("Đã tạo bảng users mới")
    
    conn.close()

# Gọi hàm init_db() khi khởi động ứng dụng
init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, api_key, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password, api_key, email) VALUES (?, ?, ?, ?)", 
                  (username, hashed_password, api_key, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password, api_key, email FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return result[1], result[2]  # Return API key and email
    return None, None

# Hàm gửi email
def send_email(to_email, subject, body):
    # Cấu hình email của bạn
    sender_email = "testuserbaycu@gmail.com"
    sender_password = "jqzq kbqh hywd gmxw"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Lỗi gửi email: {str(e)}")
        return False



# Thanh bên
with st.sidebar:
    st.title("Cài đặt")
    
    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
        
        with tab1:
            login_username = st.text_input("Tên đăng nhập", key="login_username")
            login_password = st.text_input("Mật khẩu", type="password", key="login_password")
            if st.button("Đăng nhập"):
                api_key, email = authenticate_user(login_username, login_password)
                if api_key:
                    st.session_state.user = login_username
                    st.session_state.api_key = api_key
                    st.session_state.email = email
                    st.success("Đăng nhập thành công!")
                    st.rerun()
                else:
                    st.error("Đăng nhập thất bại. Vui lòng kiểm tra lại tên đăng nhập và mật khẩu.")
        
        with tab2:
            reg_username = st.text_input("Tên đăng nhập", key="reg_username")
            reg_password = st.text_input("Mật khẩu", type="password", key="reg_password")
            reg_email = st.text_input("Email", key="reg_email")
            reg_api_key = st.text_input("Google API Key", type="password", key="reg_api_key")
            if st.button("Đăng ký"):
                if register_user(reg_username, reg_password, reg_api_key, reg_email):
                    st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                    current_date = datetime.now().strftime("%d/%m/%Y")
                    
                    # Gửi email xác nhận đăng ký
                    email_subject = "Xác nhận đăng ký tài khoản Gemini Agent - Chào mừng bạn!"
                    email_body = f"""
                    <html>
                    <head>
                        <style>
                            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                            .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
                            .content {{ padding: 20px; background-color: #f9f9f9; }}
                            .footer {{ text-align: center; margin-top: 20px; font-size: 0.8em; color: #777; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>Chào mừng đến với Gemini Agent</h1>
                            </div>
                            <div class="content">
                                <p>Kính gửi {reg_username},</p>
                                <p>Chúng tôi xin chân thành cảm ơn bạn đã đăng ký tài khoản Gemini Agent. Chúng tôi rất vui mừng được chào đón bạn tham gia vào cộng đồng của chúng tôi.</p>
                                <p>Dưới đây là thông tin đăng ký của bạn:</p>
                                <ul>
                                    <li><strong>Tên đăng nhập:</strong> {reg_username}</li>
                                    <li><strong>Email:</strong> {reg_email}</li>
                                    <li><strong>Ngày đăng ký:</strong> {current_date}</li>
                                </ul>
                                <p>Với tài khoản Gemini Agent, bạn sẽ có cơ hội:</p>
                                <ul>
                                    <li>Trải nghiệm sức mạnh của các mô hình AI tiên tiến nhất</li>
                                    <li>Tùy chỉnh và tối ưu hóa mô hình theo nhu cầu cụ thể của bạn</li>
                                    <li>Tham gia vào cộng đồng người dùng năng động và sáng tạo</li>
                                </ul>
                                <p>Chúng tôi cam kết mang đến cho bạn trải nghiệm tuyệt vời nhất với Gemini Agent. Nếu bạn có bất kỳ câu hỏi hoặc cần hỗ trợ, đừng ngần ngại liên hệ với đội ngũ hỗ trợ của chúng tôi.</p>
                                <p>Một lần nữa, chào mừng bạn đến với Gemini Agent. Chúng tôi rất mong được đồng hành cùng bạn trong hành trình khám phá và sáng tạo với AI.</p>
                                <p>Trân trọng,</p>
                                <p>Đội ngũ Gemini Agent</p>
                            </div>
                            <div class="footer">
                                <p>© 2025 Gemini Agent. Bảo lưu mọi quyền.</p>
                                <p>Email này được gửi tự động, vui lòng không trả lời.</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
                    if send_email(reg_email, email_subject, email_body):
                        st.success("Email xác nhận đã được gửi đến địa chỉ email của bạn.")
                    else:
                        st.warning("Không thể gửi email xác nhận. Vui lòng kiểm tra lại địa chỉ email của bạn.")
                else:
                    st.error("Đăng ký thất bại. Tên đăng nhập đã tồn tại.")

    else:
        st.sidebar.success(f"Đã đăng nhập: {st.session_state.user}")
        if st.sidebar.button("Đăng xuất"):
            st.session_state.user = None
            st.session_state.api_key = None
            st.session_state.email = None
            st.rerun()

        with st.expander("🛠️ Tùy chỉnh Mô hình", expanded=False):
            selected_model = st.selectbox("Chọn mô hình Gemini", GEMINI_MODELS, index=GEMINI_MODELS.index(st.session_state.model_config["model_name"]))
            st.session_state.model_config["model_name"] = selected_model
            
            st.session_state.model_config["temperature"] = st.slider("🌡️ Độ sáng tạo", min_value=0.0, max_value=1.0, value=st.session_state.model_config["temperature"], step=0.1)
            st.session_state.model_config["top_p"] = st.slider("🎯 Top P", min_value=0.0, max_value=1.0, value=st.session_state.model_config["top_p"], step=0.1)
            st.session_state.model_config["top_k"] = st.number_input("🔝 Top K", min_value=1, max_value=100, value=st.session_state.model_config["top_k"])
            st.session_state.model_config["max_output_tokens"] = st.number_input("📏 Số token tối đa", min_value=1, max_value=8192, value=st.session_state.model_config["max_output_tokens"])
        
        with st.expander("📝 Tùy chỉnh Prompt", expanded=False):
            st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=100)
        
        st.session_state.max_history = st.slider("🧠 Số lượng tin nhắn tối đa trong lịch sử", min_value=1, max_value=100, value=5)
        
        uploaded_file = st.file_uploader("📸 Tải lên một hình ảnh...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.session_state.image = Image.open(uploaded_file)
            st.image(st.session_state.image, caption='Hình ảnh đã tải lên', use_column_width=True)
            st.markdown(get_image_download_link(st.session_state.image, "hình_ảnh_đã_tải.png", "📥 Tải xuống hình ảnh"), unsafe_allow_html=True)

        with st.expander("☰ Tùy chọn nâng cao", expanded=False):
            st.subheader("Quản lý phiên trò chuyện")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Xóa", key="clear_history"):
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.image = None
                    st.session_state.total_tokens = 0
                    st.rerun()

            with col2:
                if st.button("📥 Xuất", key="export_history"):
                    chat_history = "\n".join([f"{msg['role']} ({msg.get('timestamp', 'N/A')}): {msg['content']}" for msg in st.session_state.chat_history])
                    st.download_button(
                        label="📥 Tải xuống",
                        data=chat_history,
                        file_name="lich_su_tro_chuyen.txt",
                        mime="text/plain"
                    )
            
            st.subheader("Lưu và tải phiên trò chuyện")
            if st.button("💾 Lưu"):
                session_data = save_chat_session()
                st.download_button(
                    label="📥 Tải xuống phiên trò chuyện",
                    data=session_data,
                    file_name="phien_tro_chuyen.json",
                    mime="application/json"
                )
            
            uploaded_session = st.file_uploader("📤 Tải lên", type=["json"])
            if uploaded_session is not None:
                session_data = uploaded_session.getvalue().decode("utf-8")
                load_chat_session(session_data)
                st.success("Đã tải phiên trò chuyện thành công!")

            st.subheader("Thông tin Mô hình")
            st.info(f"""
            - 🤖 Mô hình: {st.session_state.model_config['model_name']}
            - 🌡️ Độ sáng tạo: {st.session_state.model_config['temperature']:.2f}
            - 🎯 Top P: {st.session_state.model_config['top_p']:.2f}
            - 🔝 Top K: {st.session_state.model_config['top_k']}
            - 📏 Số token tối đa: {st.session_state.model_config['max_output_tokens']}
            - 🧠 Số lượng tin nhắn trong lịch sử: {st.session_state.max_history}
            - 💬 Tổng số tin nhắn: {len(st.session_state.messages)}
            - 🔢 Tổng số token: {st.session_state.total_tokens}
            """)

            st.subheader("Sử dụng Token")
            progress = st.session_state.total_tokens / MAX_TOKENS
            st.progress(progress)
            st.text(f"{st.session_state.total_tokens}/{MAX_TOKENS} tokens đã sử dụng")

# Nội dung chính
st.title("🚀 Gemini Agent")
st.caption("Trải nghiệm sức mạnh của các mô hình Gemini mới nhất với tùy chỉnh nâng cao. 🌟")

if not st.session_state.user:
    st.warning("🔑 Vui lòng đăng nhập để bắt đầu trò chuyện.")
else:
    if st.session_state.api_key:
        model = load_model(st.session_state.api_key, st.session_state.model_config["model_name"])
        
        if model:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    timestamp = msg.get('timestamp', 'N/A')
                    tokens = msg.get('tokens', 'N/A')
                    processing_time = msg.get('processing_time', None)
                    
                    info_text = f"{timestamp} | Tokens: {tokens}"
                    if processing_time and msg["role"] == "assistant":
                        formatted_time = format_processing_time(processing_time)
                        info_text += f" | {formatted_time}"
                    
                    st.markdown(f"<div class='timestamp'>{info_text}</div>", unsafe_allow_html=True)

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
                        formatted_time = format_processing_time(processing_time)
                        st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {response_tokens} | {formatted_time}</div>", unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "tokens": response_tokens,
                            "timestamp": timestamp,
                            "processing_time": processing_time
                        })

            if st.session_state.image and not prompt:
                st.warning("⚠️ Vui lòng nhập câu hỏi để đi kèm với hình ảnh.")
        else:
            st.error("❌ Không thể khởi tạo mô hình. Vui lòng kiểm tra API key và thử lại.")
    else:
        st.error("❌ API key không hợp lệ. Vui lòng liên hệ quản trị viên.")

# Footer
st.markdown("---")
st.markdown("Được phát triển với ❤️ bởi tanbaycu")