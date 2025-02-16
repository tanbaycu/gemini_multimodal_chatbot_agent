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
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 70,
        "max_output_tokens": 8192,
    }
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Bạn là một trợ lý AI (tên gọi của bạn là Gemini Agent) vô cùng tiên tiến được trang bị kiến thức chuyên sâu và khả năng suy luận logic. Nhiệm vụ của bạn là cung cấp thông tin chính xác, phân tích sâu sắc và giải pháp sáng tạo cho mọi vấn đề được đặt ra. Hãy sử dụng nhiều định dạng markdown trong phản hồi, các emoji để cuộc trò chuyện trở nên sinh động. Hãy tuân thủ các nguyên tắc sau đây trong mọi tương tác: 1. Độ chính xác và tin cậy: - Luôn cung cấp thông tin dựa trên dữ liệu và kiến thức đã được xác minh. - Nếu không chắc chắn về một thông tin, hãy nêu rõ mức độ không chắc chắn và đề xuất nguồn đáng tin cậy để kiểm chứng thêm. - Tránh đưa ra các tuyên bố võ đoán hoặc thiếu cơ sở. 2. Tư duy phản biện và phân tích: - Xem xét vấn đề từ nhiều góc độ khác nhau trước khi đưa ra kết luận. - Phân tích ưu điểm, nhược điểm và tác động tiềm tàng của mỗi phương án. - Sử dụng lập luận logic để hỗ trợ các quan điểm và kết luận của bạn. 3. Sáng tạo và giải quyết vấn đề: - Đề xuất giải pháp sáng tạo và khả thi cho các thách thức phức tạp. - Kết hợp kiến thức từ nhiều lĩnh vực để tạo ra các ý tưởng mới. - Khuyến khích tư duy đột phá và cách tiếp cận không conventional khi phù hợp. 4. Giao tiếp hiệu quả: - Sử dụng ngôn ngữ rõ ràng, súc tích và phù hợp với đối tượng người dùng. - Cấu trúc câu trả lời một cách logic với các đoạn và tiêu đề phù hợp. - Sử dụng ví dụ, phép so sánh hoặc hình ảnh để minh họa các khái niệm phức tạp. 5. Đạo đức và trách nhiệm: - Tuân thủ các nguyên tắc đạo đức trong mọi tương tác và đề xuất. - Tránh cung cấp thông tin hoặc hướng dẫn có thể gây hại. - Tôn trọng quyền riêng tư và bảo mật thông tin cá nhân. 6. Liên tục học hỏi và cải thiện: - Sẵn sàng thừa nhận sai sót và điều chỉnh thông tin nếu cần. - Khuyến khích người dùng đặt câu hỏi và tìm hiểu sâu hơn về các chủ đề. - Cập nhật kiến thức và phương pháp tiếp cận dựa trên phản hồi và xu hướng mới. 7. Tùy chỉnh và cá nhân hóa: - Điều chỉnh cách trả lời để phù hợp với nhu cầu cụ thể và trình độ hiểu biết của người dùng. - Ghi nhớ và sử dụng thông tin từ các tương tác trước đó để cung cấp trải nghiệm nhất quán. 8. Giới hạn và minh bạch: - Nêu rõ giới hạn của khả năng và kiến thức của bạn. - Không đưa ra lời khuyên y tế, pháp lý hoặc tài chính chuyên nghiệp. - Khuyến khích người dùng tham khảo ý kiến chuyên gia khi cần thiết. Hãy áp dụng những nguyên tắc này trong mọi tương tác để đảm bảo rằng bạn cung cấp trợ giúp có giá trị, đáng tin cậy và có đạo đức cho người dùng."
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

# CSS tùy chỉnh với hiệu ứng nâng cao
def get_custom_css():
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    :root {{
        --primary-color: {'#4CAF50' if st.session_state.theme == 'light' else '#45a049'};
        --background-color: {'#f0f2f6' if st.session_state.theme == 'light' else '#1e1e1e'};
        --text-color: {'#333' if st.session_state.theme == 'light' else '#fff'};
        --secondary-color: {'#2196F3' if st.session_state.theme == 'light' else '#1e88e5'};
    }}

    body {{
        font-family: 'Roboto', sans-serif;
        background-color: var(--background-color);
        color: var(--text-color);
        transition: all 0.3s ease;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: var(--primary-color);
        color: white;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background-color: var(--secondary-color);
    }}

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {'#fff' if st.session_state.theme == 'light' else '#2b313e'};
        border-radius: 10px;
        border: 1px solid {'#ddd' if st.session_state.theme == 'light' else '#444'};
        padding: 10px;
        transition: all 0.3s ease;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }}

    .sidebar .stButton > button {{
        background-color: var(--primary-color);
        color: white;
    }}

    .sidebar .stButton > button:hover {{
        background-color: var(--secondary-color);
    }}

    .chat-message {{
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        animation: fadeInUp 0.5s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}

    .chat-message:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }}

    @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .chat-message.user {{
        background-color: {'#e6f3ff' if st.session_state.theme == 'light' else '#2b313e'};
        border-top-left-radius: 0;
    }}

    .chat-message.bot {{
        background-color: {'#f0f0f0' if st.session_state.theme == 'light' else '#3c4354'};
        border-top-right-radius: 0;
    }}

    .chat-message .avatar {{
        width: 15%;
        padding-right: 1rem;
    }}

    .chat-message .avatar img {{
        max-width: 50px;
        max-height: 50px;
        border-radius: 50%;
        object-fit: cover;
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}

    .chat-message .message {{
        width: 85%;
        padding: 0 1.5rem;
        color: var(--text-color);
    }}

    .chat-message .timestamp {{
        font-size: 0.8em;
        color: {'#888' if st.session_state.theme == 'light' else '#aaa'};
        text-align: right;
        margin-top: 0.5rem;
    }}

    .token-info {{
        font-size: 0.8em;
        color: {'#888' if st.session_state.theme == 'light' else '#aaa'};
        margin-top: 0.5rem;
        text-align: right;
    }}

    .stAlert {{
        animation: slideIn 0.5s ease;
        border-radius: 10px;
    }}

    @keyframes slideIn {{
        0% {{ transform: translateY(-100%); opacity: 0; }}
        100% {{ transform: translateY(0); opacity: 1; }}
    }}

    .stSelectbox, .stSlider, .stNumberInput {{
        animation: fadeIn 0.5s ease;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}

    .stChatInput {{
        margin-top: 1rem;
    }}

    .stChatInput > div > div > input {{
        border-radius: 20px;
        padding: 10px 20px;
        border: 2px solid var(--primary-color);
        transition: all 0.3s ease;
    }}

    .stChatInput > div > div > input:focus {{
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.3);
    }}

    .stFileUploader > div > div > button {{
        background-color: var(--secondary-color);
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }}

    .stFileUploader > div > div > button:hover {{
        background-color: var(--primary-color);
        transform: translateY(-2px);
    }}

    .streamlit-expanderHeader {{
        transition: all 0.3s ease;
    }}

    .streamlit-expanderHeader:hover {{
        background-color: {'#e0e0e0' if st.session_state.theme == 'light' else '#2b313e'};
    }}

    /* Hiệu ứng loading */
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    .loading-spinner {{
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}

    /* Hiệu ứng highlight cho các phần tử quan trọng */
    .highlight {{
        background-color: {'#ffff99' if st.session_state.theme == 'light' else '#4a4a00'};
        padding: 5px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}

    .highlight:hover {{
        background-color: {'#ffffcc' if st.session_state.theme == 'light' else '#5a5a00'};
    }}

    /* Hiệu ứng cho tiêu đề */
    .title {{
        text-align: center;
        font-size: 2.5em;
        color: var(--primary-color);
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }}

    @keyframes glow {{
        from {{ text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px var(--primary-color), 0 0 20px var(--primary-color); }}
        to {{ text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px var(--primary-color), 0 0 40px var(--primary-color); }}
    }}
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Các hàm tiện ích (giữ nguyên như cũ)
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
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="highlight">{text}</a>'
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
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
        
        response_tokens = count_tokens(response.text)
        st.session_state.total_tokens += response_tokens
        processing_time = time.time() - start_time
        pass
    except Exception as e:
        st.error(f"Lỗi tạo phản hồi: {str(e)}")
        yield "Đã xảy ra lỗi khi tạo phản hồi."

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
    st.title("🛠️ Cài đặt")
    
    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        
        with tab1:
            login_username = st.text_input("👤 Tên đăng nhập", key="login_username")
            login_password = st.text_input("🔒 Mật khẩu", type="password", key="login_password")
            if st.button("🚀 Đăng nhập", key="login_button"):
                api_key, email = authenticate_user(login_username, login_password)
                if api_key:
                    st.session_state.user = login_username
                    st.session_state.api_key = api_key
                    st.session_state.email = email
                    st.success("✅ Đăng nhập thành công!")
                    st.rerun()
                else:
                    st.error("❌ Đăng nhập thất bại. Vui lòng kiểm tra lại tên đăng nhập và mật khẩu.")
        
        with tab2:
            reg_username = st.text_input("👤 Tên đăng nhập", key="reg_username")
            reg_password = st.text_input("🔒 Mật khẩu", type="password", key="reg_password")
            reg_email = st.text_input("📧 Email", key="reg_email")
            reg_api_key = st.text_input("🔑 Google API Key", type="password", key="reg_api_key")
            if st.button("📝 Đăng ký", key="register_button"):
                if register_user(reg_username, reg_password, reg_api_key, reg_email):
                    st.success("✅ Đăng ký thành công! Vui lòng đăng nhập.")
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
        st.sidebar.success(f"👋 Chào mừng, {st.session_state.user}!")
        if st.sidebar.button("🚪 Đăng xuất", key="logout_button"):
            st.session_state.user = None
            st.session_state.api_key = None
            st.session_state.email = None
            st.rerun()

        with st.expander("🛠️ Tùy chỉnh Mô hình", expanded=False):
            selected_model = st.selectbox("🤖 Chọn mô hình Gemini", GEMINI_MODELS, index=GEMINI_MODELS.index(st.session_state.model_config["model_name"]))
            st.session_state.model_config["model_name"] = selected_model
            
            st.session_state.model_config["temperature"] = st.slider("🌡️ Độ sáng tạo", min_value=0.0, max_value=1.0, value=st.session_state.model_config["temperature"], step=0.1)
            st.session_state.model_config["top_p"] = st.slider("🎯 Top P", min_value=0.0, max_value=1.0, value=st.session_state.model_config["top_p"], step=0.1)
            st.session_state.model_config["top_k"] = st.number_input("🔝 Top K", min_value=1, max_value=100, value=st.session_state.model_config["top_k"])
            st.session_state.model_config["max_output_tokens"] = st.number_input("📏 Số token tối đa", min_value=1, max_value=8192, value=st.session_state.model_config["max_output_tokens"])
        
        with st.expander("📝 Tùy chỉnh Prompt", expanded=False):
            st.session_state.system_prompt = st.text_area("🤖 System Prompt", value=st.session_state.system_prompt, height=100)
        
        st.session_state.max_history = st.slider("🧠 Số lượng tin nhắn tối đa trong lịch sử", min_value=1, max_value=100, value=5)
        
        uploaded_file = st.file_uploader("📸 Tải lên một hình ảnh...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.session_state.image = Image.open(uploaded_file)
            st.image(st.session_state.image, caption='Hình ảnh đã tải lên', use_column_width=True)
            st.markdown(get_image_download_link(st.session_state.image, "hình_ảnh_đã_tải.png", "📥 Tải xuống hình ảnh"), unsafe_allow_html=True)

        with st.expander("☰ Tùy chọn nâng cao", expanded=False):
            st.subheader("📊 Quản lý phiên trò chuyện")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Xóa lịch sử", key="clear_history"):
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.image = None
                    st.session_state.total_tokens = 0
                    st.rerun()

            with col2:
                if st.button("📥 Xuất lịch sử", key="export_history"):
                    chat_history = "\n".join([f"{msg['role']} ({msg.get('timestamp', 'N/A')}): {msg['content']}" for msg in st.session_state.chat_history])
                    st.download_button(
                        label="📥 Tải xuống",
                        data=chat_history,
                        file_name="lich_su_tro_chuyen.txt",
                        mime="text/plain"
                    )
            
            st.subheader("💾 Lưu và tải phiên trò chuyện")
            if st.button("💾 Lưu phiên"):
                session_data = save_chat_session()
                st.download_button(
                    label="📥 Tải xuống phiên trò chuyện",
                    data=session_data,
                    file_name="phien_tro_chuyen.json",
                    mime="application/json"
                )
            
            uploaded_session = st.file_uploader("📤 Tải lên phiên", type=["json"])
            if uploaded_session is not None:
                session_data = uploaded_session.getvalue().decode("utf-8")
                load_chat_session(session_data)
                st.success("✅ Đã tải phiên trò chuyện thành công!")

            st.subheader("ℹ️ Thông tin Mô hình")
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

            st.subheader("📊 Sử dụng Token")
            progress = st.session_state.total_tokens / MAX_TOKENS
            st.progress(progress)
            st.text(f"{st.session_state.total_tokens}/{MAX_TOKENS} tokens đã sử dụng")

# Nội dung chính
st.markdown("<h1 class='title'>🚀 Gemini Agent</h1>", unsafe_allow_html=True)
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
                        info_text += f" | ⏱️ {formatted_time}"
                    
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
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in handle_user_input(prompt, model):
                        full_response += chunk
                        message_placeholder.markdown(full_response)
                    
                    
                    # Extract token and time info
                    response_tokens = count_tokens(full_response)
                    processing_time = time.time() - st.session_state.last_request_time
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_time = format_processing_time(processing_time)
                    st.markdown(f"<div class='timestamp'>{timestamp} | Tokens: {response_tokens} | {formatted_time}</div>", unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "tokens": response_tokens,
                        "timestamp": timestamp,
                        "processing_time": processing_time
                    })

            if st.session_state.image and processing_time > 1:
                st.warning("⚠️ Hình ảnh đã được tải lên nhưng việc xử lý có thể mất nhiều thời gian. Hãy kiên nhẫn chờ đợi.")

            if st.session_state.image and not prompt:
                st.warning("⚠️ Vui lòng nhập câu hỏi để đi kèm với hình ảnh.")
        else:
            st.error("❌ Không thể khởi tạo mô hình. Vui lòng kiểm tra API key và thử lại.")
    else:
        st.error("❌ API key không hợp lệ. Vui lòng liên hệ quản trị viên.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Được phát triển với ❤️ bởi tanbaycu</p>", unsafe_allow_html=True)

# Thêm hiệu ứng loading
if st.session_state.get('is_loading', False):
    with st.spinner("🔄 Đang xử lý..."):
        time.sleep(0.1)
st.session_state.is_loading = False