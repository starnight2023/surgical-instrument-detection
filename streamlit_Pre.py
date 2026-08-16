import os
import io
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# 1. 设置页面标题
st.set_page_config(page_title="手术器械检测系统", layout="wide")
st.title("手术器械检测")

# 自定义 CSS 样式：将按钮渲染为醒目的红色
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    div.stButton > button:hover {
        background-color: #FF2B2B;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# 定义后端 API 地址
API_URL = "http://fastapi-backend:8000/predict"

# 默认置信度过滤阈值（既然前台不再显示滑动条，我们直接在后台写死一个默认值）
CONF_THRESHOLD = 0.15

# 2. 初始化会话状态，用于控制“开始检测”按钮的显示与结果持久化
if "detected" not in st.session_state:
    st.session_state.detected = False
if "result_image" not in st.session_state:
    st.session_state.result_image = None
if "detected_count" not in st.session_state:
    st.session_state.detected_count = 0
if "last_image_key" not in st.session_state:
    st.session_state.last_image_key = None

# --- 示例图片配置 ---
SAMPLE_DIR = "sample_image"
sample_files = []
if os.path.exists(SAMPLE_DIR):
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sample_files.sort()

# --- 主页面布局 ---
uploaded_file = st.file_uploader("请上传一张手术图片", type=["jpg", "jpeg", "png"])

selected_sample = None
if sample_files:
    selected_sample = st.selectbox("或者选择以下图片：", ["-- 请选择以下图片 --"] + sample_files)

# 确定当前处理的图像与 key、字节流
image = None
image_data = None
current_image_key = None

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data))
    current_image_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
elif selected_sample and selected_sample != "-- 请选择以下图片 --":
    image_path = os.path.join(SAMPLE_DIR, selected_sample)
    image = Image.open(image_path)
    # 将本地图片读取为字节流以便发送给 FastAPI
    with open(image_path, "rb") as f:
        image_data = f.read()
    current_image_key = f"sample_{selected_sample}"

# 如果更换了图片，自动重置状态，重新显示检测按钮
if current_image_key != st.session_state.last_image_key:
    st.session_state.detected = False
    st.session_state.result_image = None
    st.session_state.detected_count = 0
    st.session_state.last_image_key = current_image_key


# --- 展示区域 ---
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图片")
        # 升级为官方最新的 width="stretch" 适配，彻底消除终端警告
        st.image(image, width="stretch")

    with col2:
        if st.session_state.detected and st.session_state.result_image:
            st.subheader(f"检测结果 (发现 {st.session_state.detected_count} 个目标)")
            st.image(st.session_state.result_image, width="stretch")
        else:
            st.subheader("检测结果 (未开始)")
            st.info("请在下方点击“开始检测”获取器械识别结果。")

    # 仅在未检测时显示红色的“开始检测”按钮
    if not st.session_state.detected:
        if st.button("开始检测", type="primary"):
            with st.spinner("正在请求 AI 模型进行分析..."):
                try:
                    # 发送请求给 FastAPI
                    files = {"file": ("image.jpg", image_data, "image/jpeg")}
                    response = requests.post(API_URL, files=files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        
                        # 检查后端是否返回了异常（比如 NumPy 不兼容等报错）
                        if "error" in result:
                            st.error(f"算法服务报错: {result['error']}")
                        else:
                            detections = result.get("detections", [])

                            # 拷贝一份原始图像利用 PIL 进行画框，避免破坏原始图像展示
                            draw_image = image.copy()
                            draw = ImageDraw.Draw(draw_image)
                            
                            try:
                                font = ImageFont.truetype("arial.ttf", 24)
                            except:
                                font = ImageFont.load_default()

                            count = 0
                            for item in detections:
                                score = item['confidence']
                                # 过滤置信度低于设定的阈值的目标
                                if score < CONF_THRESHOLD:
                                    continue

                                count += 1
                                box = item['bbox']
                                x_min, y_min = box['x_min'], box['y_min']
                                x_max, y_max = box['x_max'], box['y_max']
                                label = f"{item['class_name']} ({score:.2f})"

                                # 画矩形框 (红色，宽度3)
                                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                                # 画文字背景
                                text_bbox = draw.textbbox((x_min, y_min), label, font=font)
                                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
                                # 画文字
                                draw.text((x_min, y_min), label, fill="white", font=font)

                            # 将画完框的图像以及目标计数保存在 Session 状态中
                            st.session_state.result_image = draw_image
                            st.session_state.detected_count = count
                            st.session_state.detected = True
                            st.rerun()  # 触发重绘，按钮会自动消失，右侧展示画框结果

                    else:
                        st.error(f"服务器报错，状态码: {response.status_code}")

                except Exception as e:
                    # 如果真的无法连接到 FastAPI，则在此处捕获
                    st.error(f"无法连接到后端服务: {e}")