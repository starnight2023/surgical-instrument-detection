from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import time
from fastapi.middleware.cors import CORSMiddleware  # 给 FastAPI 加上“允许跨域”的通行证。

# 导入 bubbliiiing 库中的 YOLO 类
from yolo import YOLO

app = FastAPI(title="手术器械检测 API")

# 配置 CORS，进行允许跨域（即实现不同域名网站之间的通信，使streamlit能够访问到fastapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问，生产环境可以指定具体域名，或者指定允许streamlit的 ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST等)
    allow_headers=["*"],  # 允许所有 Header
)

# 1. 全局区域：初始化模型（这样不用每次请求都重新加载模型，速度快。如果放在函数里，每次请求都加载）模型驻留在内存/显存中
# 确保此时你的 model_path 配置是正确的
print("正在加载 YOLO 模型...")
yolo_model = YOLO()
print("模型加载完成！")

@app.get("/")
def read_root():
    return {"message": "Surgical Instrument Detection API is running!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 2. 校验文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 3. 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 4. 调用我们在 yolo.py 中修改好的 API 方法
        start_time = time.time()
        results = yolo_model.detect_image_api(image)
        end_time = time.time()

        # 5. 返回 JSON 结果
        return {
            "filename": file.filename,
            "inference_time": f"{end_time - start_time:.4f}s",
            "detections": results,
            "count": len(results)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，host 0.0.0.0 允许局域网访问
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 0.0.0.0：是指 “监听这台电脑上所有的网卡接口”
