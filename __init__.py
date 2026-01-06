import torch
import numpy as np
import requests
import json
import base64
import io
from PIL import Image

# ==========================================
# 核心节点类：大香蕉Pro (文档修正版)
# ==========================================
class BigBananaProNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "Enter Google API Key"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A futuristic city with flying cars, cinematic lighting, 4k, masterpiece"}),
                "model_name": ("STRING", {"multiline": False, "default": "gemini-3-pro-image-preview"}),
                "base_url": ("STRING", {"multiline": False, "default": "https://generativelanguage.googleapis.com/v1beta/models"}),
                
                # 修正1：分辨率参数名回归文档标准的 "1K/2K/4K"
                "quality_grade": (["1K", "2K", "4K"], {"default": "1K"}),
                
                # 修正2：长宽比使用标准枚举值
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Banana"

    def tensor_to_base64(self, image_tensor):
        img_tensor = image_tensor[0]
        i = 255. * img_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=95) 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate_image(self, api_key, prompt, model_name, base_url, quality_grade, aspect_ratio, reference_images=None):
        if not api_key:
            raise ValueError("API Key is required / 需要填写 API Key")

        # 1. URL 构建
        base_url = base_url.rstrip("/")
        if ":generateContent" not in base_url and ":predict" not in base_url:
            url = f"{base_url}/{model_name}:generateContent"
        else:
            url = base_url

        params = {"key": api_key}
        headers = {"Content-Type": "application/json"}

        # 2. Payload 构建
        parts = [{"text": prompt}]

        # 垫图处理
        if reference_images is not None:
            for idx in range(reference_images.shape[0]):
                single_img = reference_images[idx:idx+1]
                b64_str = self.tensor_to_base64(single_img)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_str
                    }
                })

        # 3. 严格遵循文档的 Image Config
        # 文档地址: https://ai.google.dev/gemini-api/docs/image-generation
        # imageConfig 包含: 
        #   - aspectRatio: "16:9", "1:1", etc.
        #   - imageSize: "1K", "2K", "4K" (注意参数名是 imageSize，不是 resolution)
        
        image_config = {
            "aspectRatio": aspect_ratio,
            "imageSize": quality_grade  # 这里映射到 imageSize
        }

        generation_config = {
            "temperature": 1.0,
            "responseModalities": ["IMAGE"], 
            "imageConfig": image_config
        }

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config
        }

        print(f"--- BigBananaPro (Strict Docs) ---")
        print(f"Model: {model_name} | Size: {quality_grade} | AR: {aspect_ratio}")
        
        try:
            response = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request Failed: {e}"
            if 'response' in locals() and response is not None:
                error_msg += f"\nServer Response: {response.text}"
            raise RuntimeError(error_msg)

        # 4. 解析图片
        output_images = []
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                feedback = response_json.get("promptFeedback", {})
                raise RuntimeError(f"No candidates. Feedback: {feedback}\nResponse: {response_json}")

            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data:
                        b64_data = inline_data.get("data")
                        if b64_data:
                            img_data = base64.b64decode(b64_data)
                            img = Image.open(io.BytesIO(img_data))
                            output_images.append(img)
                    elif "text" in part:
                         print(f"Info: Model returned text: {part['text'][:50]}...")

        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}")

        if not output_images:
            raise RuntimeError("API success but no image returned.")

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
    "BigBananaProNode": "大香蕉Pro (Gemini 3 Docs版)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
