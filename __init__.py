import torch
import numpy as np
import requests
import json
import base64
import io
import urllib.request  # 引入这个库用来抓取系统代理
from PIL import Image

# ==========================================
# 核心节点类：大香蕉Pro (中文 + 系统代理增强版)
# ==========================================
class BigBananaProNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": "", "placeholder": "输入 Google API Key"}),
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A futuristic city with flying cars, cinematic lighting, 4k, masterpiece"}),
                "模型名称": ("STRING", {"multiline": False, "default": "gemini-3-pro-image-preview"}),
                "API地址": ("STRING", {"multiline": False, "default": "https://generativelanguage.googleapis.com/v1beta/models"}),
                
                "画质等级": (["1K", "2K", "4K"], {"default": "1K"}),
                "长宽比": (["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "5:4", "4:5", "未指定(Free)"], {"default": "1:1"}),
                
                # 新增代理设置
                "代理地址": ("STRING", {"multiline": False, "default": "", "placeholder": "留空自动用系统代理，或填 http://127.0.0.1:7890"}),
            },
            "optional": {
                "参考图": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate_image"
    CATEGORY = "Banana"

    def tensor_to_base64(self, img_tensor):
        i = 255. * img_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=95) 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate_image(self, API密钥, 提示词, 模型名称, API地址, 画质等级, 长宽比, 代理地址, 参考图=None):
        
        # 变量映射
        api_key = API密钥
        prompt = 提示词
        model_name = 模型名称
        base_url = API地址
        quality_grade = 画质等级
        aspect_ratio_val = 长宽比
        reference_images = 参考图
        proxy_url = 代理地址.strip() # 去除首尾空格

        if not api_key:
            raise ValueError("API Key is required / 请输入 API 密钥")

        # =================================================
        # 代理设置逻辑 (Proxy Setup)
        # =================================================
        proxies = None
        
        if proxy_url:
            # 1. 如果用户手动填了代理，优先使用
            print(f"--- 使用手动代理: {proxy_url} ---")
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
        else:
            # 2. 如果留空，尝试抓取系统代理
            system_proxies = urllib.request.getproxies()
            if system_proxies:
                print(f"--- 检测到系统代理: {system_proxies} ---")
                proxies = system_proxies
            else:
                print("--- 未检测到系统代理，将直连 (Direct Connect) ---")

        # =================================================
        # 请求逻辑
        # =================================================
        base_url = base_url.rstrip("/")
        if ":generateContent" not in base_url and ":predict" not in base_url:
            url = f"{base_url}/{model_name}:generateContent"
        else:
            url = base_url

        params = {"key": api_key}
        headers = {"Content-Type": "application/json"}

        # Payload
        parts = [{"text": prompt}]

        if reference_images is not None:
            batch_size = reference_images.shape[0]
            print(f"--- 正在处理 {batch_size} 张参考图 ---")
            for idx in range(batch_size):
                single_img = reference_images[idx]
                b64_str = self.tensor_to_base64(single_img)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_str
                    }
                })

        # Image Config
        image_config = {
            "imageSize": quality_grade
        }
        if "Free" not in aspect_ratio_val:
            image_config["aspectRatio"] = aspect_ratio_val

        generation_config = {
            "temperature": 1.0,
            "responseModalities": ["IMAGE"], 
            "imageConfig": image_config
        }

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config
        }

        print(f"--- BigBananaPro Request ---")
        print(f"Model: {model_name} | Size: {quality_grade} | AR: {aspect_ratio_val}")
        
        try:
            # 发送请求时带上 proxies 参数
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=payload, 
                proxies=proxies, # 关键点：应用代理
                timeout=120
            )
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Failed (网络错误): {e}"
            if proxies:
                error_msg += f"\n当前使用的代理: {proxies}"
            if 'response' in locals() and response is not None:
                error_msg += f"\nServer Response: {response.text}"
            raise RuntimeError(error_msg)

        # 解析结果
        output_images = []
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                feedback = response_json.get("promptFeedback", {})
                raise RuntimeError(f"No candidates. Feedback: {feedback}\nResponse: {response_json}")

            for candidate in candidates:
                content = candidate.get("content", {})
                res_parts = content.get("parts", [])
                for part in res_parts:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data:
                        b64_data = inline_data.get("data")
                        if b64_data:
                            img_data = base64.b64decode(b64_data)
                            img = Image.open(io.BytesIO(img_data))
                            output_images.append(img)
                    elif "text" in part:
                         print(f"Info: Text returned: {part['text'][:50]}...")

        except Exception as e:
            raise RuntimeError(f"解析响应失败: {e}")

        if not output_images:
            raise RuntimeError("API 请求成功，但未返回图片数据。")

        output_tensors = []
        for img in output_images:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            output_tensors.append(torch.from_numpy(img_np))

        return (torch.stack(output_tensors),)

# ==========================================
# 注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "BigBananaProNode": BigBananaProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigBananaProNode": "大香蕉Pro (中文+代理版)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
