import yaml
import csv
import re

def categorize_prompts(yaml_path, csv_path):
    print(f"正在读取定制配置文件: {yaml_path}")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件，请检查路径是否正确: {yaml_path}")
        return

    # 专门为你这份数据集定制的精准分类词库
    categories = {
        "Animals & Creatures": ["dog", "cat", "bird", "lion", "tiger", "fish", "turtle", "animal", "wolf", "horse", "elephant", "monkey", "rabbit", "eagle", "ant", "cow", "shiba", "inu", "creature"],
        "Urban & Vehicles": ["car", "tractor", "koeniggseg", "helicopter", "spaceship", "train", "vehicle", "shop", "apartment", "city", "skyscraper", "room", "castle", "bakery", "street", "building", "cyberpunk", "clinic"],
        "Human & Characters": ["man", "woman", "girl", "boy", "person", "people", "crowd", "face", "portrait", "soldier", "knight", "human", "guy", "lady", "character", "kid", "paul", "emma", "batman", "chef", "student", "dj", "vampire", "demoness", "model", "artist"],
        "Nature & Landscapes": ["mountain", "lake", "ocean", "beach", "forest", "tree", "river", "sky", "cloud", "rain", "snow", "landscape", "village", "cityscape", "nature", "flower", "water", "desert", "space", "meteorite", "meteor", "blossom", "storm", "flood", "sunset", "dusk", "stone", "sea", "underwater", "planet", "globe", "map"],
        "Objects & Abstract": ["food", "bottle", "logo", "text", "light", "fire", "abstract", "computer", "phone", "vase", "fragrance", "perfume", "schedule", "mirage", "coin", "mushroom", "timeline", "money", "app"]
    }

    results = []

    for item in data:
        seed = item.get('seed')
        prompt = item.get('prompt')[0].lower()
        
        assigned_category = "Other/Mixed"
        
        # 匹配逻辑：优先识别动物和城市场景，最后兜底人物和自然
        for cat_name, keywords in categories.items():
            # 使用 \b 进行单词边界匹配，s? 兼容复数形式
            if any(re.search(r'\b' + kw + r's?\b', prompt) for kw in keywords):
                assigned_category = cat_name
                break
                
        # 去掉长长的画质后缀，保持 CSV 表格干净易读
        clean_prompt = prompt.split('masterpiece')[0].strip(' ,')
                
        results.append({
            "Seed": seed,
            "Category": assigned_category,
            "Prompt": clean_prompt[:80] + "..." if len(clean_prompt) > 80 else clean_prompt
        })

    # 将分类结果写入 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Seed", "Category", "Prompt"])
        writer.writeheader()
        writer.writerows(results)

    # 在终端打印直观的统计结果
    print("\n✅ 定制分类完成！这 100 个视频的分布数据如下：")
    counts = {}
    for r in results:
        counts[r['Category']] = counts.get(r['Category'], 0) + 1
        
    # 按数量从大到小排序打印
    for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {cat}: {count} 个视频")
        
    print(f"\n分类映射表已保存至: {csv_path}")

if __name__ == "__main__":
    # 使用你服务器上的绝对路径
    yaml_file = "/data/yzj/animate1/AnimateDiff/configs/prompts/benchmark/100_vidprom_absolute_final.yaml"
    csv_file = "/data/yzj/animate1/AnimateDiff/configs/prompts/benchmark/benchmark_categories_custom.csv"
    
    categorize_prompts(yaml_file, csv_file)