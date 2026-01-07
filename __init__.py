import torch
import numpy as np
import io
import os
import sys
from PIL import Image

# 尝试导入官方库
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("\n❌ 缺少 google-genai 库，请运行 pip install google-genai\n")
    genai = None
    types = None

# ==========================================
# 核心节点类：大香蕉Pro (隐私保护版)
# ==========================================
class BigBananaProNode:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 1. 这里留空，默认去读文件
                "API密钥": ("STRING", {"multiline": False, "default": "", "placeholder": "留空则自动读取 api_key.txt"}),
                
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A futuristic city with flying cars, cinematic lighting, 4k, masterpiece"}),
                
                # 2. 改回你要的默认值
                "模型名称": ("STRING", {"multiline": False, "default": "gemini-3-pro-image-preview"}),
                
                "画质等级": (["1K", "2K", "4K"], {"default": "1K"}),
                "长宽比": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "5:4", "4:5", "未指定(Free)"], {"default": "1:1"}),
                
                "启用Google搜索": ("BOOLEAN", {"default": False, "label_on": "开启 (Grounding)", "label_off": "关闭"}),
                "代理地址": ("STRING", {"multiline": False, "default": "", "placeholder": "例如 http://127.0.0.1:7890 (留空自动)"}),
            },
            "optional": {
                "参考图": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate_image"
    CATEGORY = "Banana"

    def tensor_to_bytes(self, img_tensor):
        i = 255. * img_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=95) 
        return buffered.getvalue()

    def get_api_key(self, input_key):
        """
        获取 API Key 的逻辑：
        1. 优先使用节点输入框里的内容。
        2. 如果输入框为空，尝试读取插件目录下的 api_key.txt。
        """
        # 情况1: 用户在UI里填了，直接用
        if input_key and input_key.strip():
            return input_key.strip()

        # 情况2: 尝试读取文件
        key_file = os.path.join(self.current_dir, "api_key.txt")
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    file_key = f.read().strip()
                    if file_key:
                        print("🔑 已从 api_key.txt 自动加载密钥", flush=True)
                        return file_key
            except Exception as e:
                print(f"⚠️ 读取 api_key.txt 失败: {e}", flush=True)
        
        return None

    def generate_image(self, API密钥, 提示词, 模型名称, 画质等级, 长宽比, 启用Google搜索, 代理地址, 参考图=None):
        
        if genai is None:
            raise ImportError("请先安装官方库: pip install google-genai")

        # 获取真实 Key
        real_api_key = self.get_api_key(API密钥)
        if not real_api_key:
            raise ValueError("❌ 未找到 API Key！\n请在节点输入框填写，或者在插件目录创建 api_key.txt 文件。")

        print(f"\n🍌 大香蕉 Pro (SDK版) 启动...", flush=True)

        # 代理设置
        http_options = None
        proxy_url =代理地址.strip()
        if not proxy_url:
            import urllib.request
            sys_proxies = urllib.request.getproxies()
            if "http" in sys_proxies: proxy_url = sys_proxies["http"]
            elif "https" in sys_proxies: proxy_url = sys_proxies["https"]
        
        if proxy_url:
            print(f"👉 使用代理: {proxy_url}", flush=True)
            http_options = types.HttpOptions(client_args={"proxy": proxy_url})

        # 初始化客户端
        client = genai.Client(api_key=real_api_key, http_options=http_options)

        # 构建内容
        contents = [types.Content(parts=[types.Part(text=提示词)])]
        if 参考图 is not None:
            print(f"📥 添加参考图: {参考图.shape[0]} 张", flush=True)
            for idx in range(参考图.shape[0]):
                img_bytes = self.tensor_to_bytes(参考图[idx])
                contents[0].parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # 工具配置
        tools = []
        if 启用Google搜索:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
            print("🌍 Google 搜索: ON", flush=True)

        image_config = {"image_size": 画质等级}
        if "Free" not in 长宽比:
            image_config["aspect_ratio"] = 长宽比

        config = types.GenerateContentConfig(
            temperature=1.0,
            tools=tools,
            response_modalities=["IMAGE"], 
            image_config=image_config
        )

        print(f"🚀 请求模型: {模型名称} ...", flush=True)
        
        try:
            response = client.models.generate_content(
                model=模型名称,
                contents=contents,
                config=config
            )
        except Exception as e:
            raise RuntimeError(f"SDK 请求失败: {e}")

        # 解析
        output_images = []
        if not response.candidates:
             raise RuntimeError(f"生成失败，无Candidates。")

        for candidate in response.candidates:
            if candidate.grounding_metadata:
                 print(f"🔍 搜索来源: {candidate.grounding_metadata.search_entry_point}", flush=True)
            for part in candidate.content.parts:
                if part.inline_data:
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    output_images.append(img)
                elif part.text:
                    print(f"💬 模型回复: {part.text[:100]}...", flush=True)

        if not output_images:
            raise RuntimeError("未返回图片。")

        output_tensors = []
        for img in output_images:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            output_tensors.append(torch.from_numpy(img_np))

        print(f"✨ 成功生成 {len(output_tensors)} 张图片", flush=True)
        return (torch.stack(output_tensors),)

NODE_CLASS_MAPPINGS = {"BigBananaOfficialNode": BigBananaProNode}
NODE_DISPLAY_NAME_MAPPINGS = {"BigBananaOfficialNode": "大香蕉Pro (官方SDK版)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
