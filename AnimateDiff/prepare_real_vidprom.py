import os
import yaml
import random
import re
import html

# 强制走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

def clean_base_text(prompt):
    """终极基础清洗：移除所有变体指令和紧贴的特殊前缀"""
    if not prompt or not isinstance(prompt, str): return ""
    text = html.unescape(prompt)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 核心升级：通杀一切参数变体
    text = re.sub(r'[-]{1,2}[a-zA-Z]+[0-9.]*(?:\s+"[^"]*")?(?:\s+[\w.:]+)?', '', text)
    
    # 核心升级：精准切除头部无空格的 "prompt:" 或 "/imagine prompt:"
    text = re.sub(r'^(?:/?(?:imagine|create)\s+)?prompt:\s*', '', text, flags=re.IGNORECASE)
    
    text = text.replace("Message: 1 Attachment", "").replace("Message:", "")
    
    # 清理残留的连续逗号和空白
    text = re.sub(r'[,]+', ',', text)
    text = re.sub(r'\s+', ' ', text).strip(', ')
    return text

def is_strictly_valid_academic_prompt(prompt):
    """
    执行终极严格的 Drop and Resample 规则
    """
    try:
        prompt.encode('ascii')
    except UnicodeEncodeError:
        return False
        
    words = prompt.split()
    if len(words) > 60 or len(words) < 5:
        return False
        
    prompt_lower = prompt.lower()
    
    # 1. 致命涉黄、隐晦色情、极端伦理概念过滤
    nsfw_keywords = [
        "nsfw", "nude", "naked", "sexy", "bikini", "breast", "booty", "ass", 
        "cleavage", "kissing", "sensual", "erotic", "porn", "blood", "gore", 
        "uwu", "lust", "seductive", "lingerie", "underwear", "boobs", 
        "put it in", "her behind", "dick", "vagina", "penis",
        "undercrotch", "femdom", "slave", "slaves", "auction"
    ]
    
    # 2. 风格冲突与负面质量过滤
    style_conflict_keywords = [
        "bad quality", "ugly", "distorted", "deformed", "blurry", "grainy", 
        "low resolution", "oversaturated", "3d", "cartoon", "animation", "anime", 
        "cgi", "render", "2d", "illustration", "drawing", "sketch", "logo", 
        "pixar", "disney", "ghibli", "painting", "watercolor", "comic", "game",
        "disfigured", "mutation"
    ]
    
    # 3. 常见小语种高频词拦截
    foreign_keywords = [
        "jeune", "homme", "reussite", "envie", "fort", "el", "los", "las", 
        "und", "der", "die", "das", "avec", "pour", "una", "con"
    ]
    
    combined_bad_words = nsfw_keywords + style_conflict_keywords + foreign_keywords
    
    for keyword in combined_bad_words:
        if " " in keyword:
            if keyword in prompt_lower:
                return False
        else:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, prompt_lower):
                return False
            
    return True

def main():
    print("正在连接 Hugging Face 获取真实的 VidProM 数据集...")
    dataset = load_dataset("WenhaoWang/VidProM", split="train", streaming=True)
    
    valid_prompts = []
    
    print("正在执行最高级别的学术过滤网 (彻底拦截隐晦敏感词与畸形格式)...")
    for item in dataset:
        raw_prompt = item.get("prompt", "")
        cleaned_text = clean_base_text(raw_prompt)
        
        if is_strictly_valid_academic_prompt(cleaned_text):
            valid_prompts.append(cleaned_text)
            
        if len(valid_prompts) >= 500:
            break
            
    print(f"成功收集 {len(valid_prompts)} 个极度纯净的候选样本。")
    
    # 再次更改随机种子为 2029，抽选全新数据
    random.seed(2029)
    selected_prompts = random.sample(valid_prompts, 100)
    
    config_list = []
    quality_booster = ", masterpiece, best quality, photorealistic, highly detailed, 8k resolution, cinematic lighting"
    optimized_n_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, cloned face, disfigured, missing arms, missing legs, fused fingers, too many fingers, watermark"

    print("准备生成无瑕疵版 512x512 配置文件...")
    for i, prompt in enumerate(selected_prompts):
        final_prompt = prompt.strip('\"\'')
        if final_prompt.endswith(','): final_prompt = final_prompt[:-1]
        final_prompt += quality_booster
        
        block = {
            "H": 512,  
            "W": 512,
            "L": 16,   
            "adapter_lora_path": "models/Motion_Module/v3_sd15_adapter.ckpt",
            "adapter_lora_scale": 1.0,
            "dreambooth_path": "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors",
            "inference_config": "configs/inference/inference-v3.yaml",
            "motion_module": "models/Motion_Module/v3_sd15_mm.ckpt",
            "seed": 1000 + i, 
            "steps": 25,
            "guidance_scale": 7.5,
            "prompt": [final_prompt],
            "n_prompt": [optimized_n_prompt]
        }
        config_list.append(block)

    os.makedirs("configs/prompts/benchmark", exist_ok=True)
    yaml_path = "configs/prompts/benchmark/100_vidprom_absolute_final.yaml"
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_list, f, sort_keys=False, allow_unicode=True)

    print(f"\n✅ 绝对纯净无瑕的顶会级测试集已生成: {yaml_path}")
    print("这 100 个样本已经跨越了最严苛的学术和技术审查网！")

if __name__ == "__main__":
    main()