import os
import subprocess
import glob
import csv
import re
import numpy as np
from tqdm import tqdm
import argparse

# ================= 工具函数 =================
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:].zfill(8)
        result.extend([int(b) for b in bits])
    return result

def calculate_accuracy(secret_gt, extracted_bits_str):
    if not extracted_bits_str:
        return 0.0
    bits_gt = str_to_bits(secret_gt)
    bits_ex = [int(b) for b in extracted_bits_str]
    min_len = min(len(bits_gt), len(bits_ex))
    if min_len == 0:
        return 0.0
    matches = sum([1 for i in range(min_len) if bits_gt[i] == bits_ex[i]])
    return matches / min_len

# ================= 主评估流水线 =================
def main(args):
    target_dir = args.dir
    crf = args.crf
    config_file = args.config
    secret_gt = args.secret_text
    
    gif_files = glob.glob(os.path.join(target_dir, "sample", "*.gif"))
    if not gif_files:
        print(f"❌ 目录 {target_dir}/sample 中没有找到 .gif 文件！")
        return

    print(f"\n🚀 [PRO 评估管线] 发现 {len(gif_files)} 个视频。开始执行 CRF={crf} 攻击与严谨数据统计...")
    codec_name = "h265" if args.codec == "libx265" else "h264"
    # 【新增防覆盖隔离】根据编码器动态生成标签
    codec_tag = "h265" if args.codec == "libx265" else "h264"

    # 根据攻击类型动态命名
    if args.attack_type == "crf":
        attack_suffix = f"{codec_tag}_crf{crf}"
    else:
        attack_suffix = f"{codec_tag}_{args.attack_type}"

    attacked_dir = os.path.join(target_dir, f"attacked_{attack_suffix}")
    os.makedirs(attacked_dir, exist_ok=True)
    csv_file = os.path.join(target_dir, f"evaluation_{attack_suffix}_results.csv")
    
    acc_raw_list = []
    acc_rs_list = []
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_Name", "Attack_Type", "Acc_Physical(Raw)", "Acc_Application(RS)", "Extracted_Text"])
        
        for gif_path in tqdm(gif_files, desc=f"Evaluating CRF={crf}"):
            filename = os.path.basename(gif_path)
            base_name = os.path.splitext(filename)[0]
            attacked_mp4 = os.path.join(attacked_dir, f"{base_name}.mp4")
            
            # 1. 实施多维物理攻击 (去除了导致解码漂移的色彩空间锁，保留 H.265 专属 SAO 防线)
            if args.attack_type == "crf":
                if args.codec == "libx265":
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', gif_path, '-c:v', 'libx265', '-crf', str(crf), '-pix_fmt', 'yuv420p','-preset', 'slow',
                        '-x265-params', 'no-sao=1:bframes=0', attacked_mp4
                    ]
                else:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', gif_path, '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p','-preset', 'slow',
                        attacked_mp4
                    ]
                    
            elif args.attack_type == "scale":
                # 缩放攻击：长宽缩小一半再放大
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', gif_path, '-vf', 'scale=iw*0.5:ih*0.5,scale=iw*2:ih*2:flags=bilinear',
                    '-c:v', args.codec, '-crf', '18', '-preset', 'slow','-pix_fmt', 'yuv420p'
                ]
                if args.codec == "libx265":
                    ffmpeg_cmd.extend(['-x265-params', 'no-sao=1:bframes=0'])
                ffmpeg_cmd.append(attacked_mp4)
                
            elif args.attack_type == "noise":
                # 噪声攻击：添加高斯白噪声
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', gif_path, '-vf', 'noise=alls=2:allf=t',
                    '-c:v', args.codec, '-crf', '18', '-preset', 'slow','-pix_fmt', 'yuv420p'
                ]
                if args.codec == "libx265":
                    ffmpeg_cmd.extend(['-x265-params', 'no-sao=1:bframes=0'])
                ffmpeg_cmd.append(attacked_mp4)

            subprocess.run(ffmpeg_cmd, check=True)
            
            total_payload_bits = (len(args.secret_text) + args.ecc_symbols) * 8
            # 2. 调用 extract.py 获取黑盒输出
            extract_cmd = [
                'python', 'extract.py',
                '--config', config_file,
                '--video_path', attacked_mp4,
                '--target-bpf', str(args.target_bpf),
                '--ecc_symbols', str(args.ecc_symbols),
                '--msg-len', str(total_payload_bits),
                '--prompt', "" # 留空以触发自动查找
            ]
            
            result = subprocess.run(extract_cmd, capture_output=True, text=True, encoding='utf-8')
            output_log = result.stdout
            
            # 3. 正则表达式提取核心数据 (复用你优美的 benchmark 逻辑)
            extracted_text = ""
            text_match = re.search(r"提取结果:\s*(.*)", output_log)
            if text_match:
                extracted_text = text_match.group(1).strip()
                
            raw_bits_str = ""
            bits_match = re.search(r"RAW_BITS:([01]+)", output_log)
            if bits_match:
                raw_bits_str = bits_match.group(1).strip()
                
            corrected_bits_str = ""
            corr_match = re.search(r"CORRECTED_BITS:([01]+)", output_log)
            if corr_match:
                corrected_bits_str = corr_match.group(1).strip()
                
            # 4. 严谨的比特级数学计算
            acc_raw = calculate_accuracy(secret_gt, raw_bits_str)
            acc_rs = calculate_accuracy(secret_gt, corrected_bits_str) if corrected_bits_str else 0.0
            
            # 记录数据
            acc_raw_list.append(acc_raw)
            acc_rs_list.append(acc_rs)
            safe_text = repr(extracted_text)[1:-1]
            
            # 【修复写入标签】：确保 CSV 里面精准记录是 SCALE、NOISE 还是 CRF
            attack_label = f"{codec_tag.upper()}_CRF{crf}" if args.attack_type == "crf" else f"{codec_tag.upper()}_{args.attack_type.upper()}"
            writer.writerow([filename, attack_label, f"{acc_raw:.4f}", f"{acc_rs:.4f}", safe_text])
            f.flush()

    # 5. 最终统计输出
    mean_raw_acc = np.mean(acc_raw_list) * 100
    mean_rs_acc = np.mean(acc_rs_list) * 100
    
    print(f"\n==================================================")
    print(f"✅ 文件夹 {os.path.basename(target_dir)} 评估完成！")
    print(f"📊 汇总 CSV 报表已保存至: {csv_file}")
    print(f"👉 物理层平均提取准确率 (纠错前): {mean_raw_acc:.2f}%")
    print(f"👉 应用层平均提取准确率 (RS纠错后): {mean_rs_acc:.2f}%")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="生成的带有时间戳的目录路径")
    parser.add_argument("--config", type=str, default="configs/prompts/benchmark/100_vidprom_absolute_final.yaml")
    parser.add_argument("--secret-text", type=str, default="My confidential message!")
    parser.add_argument("--crf", type=int, default=30, help="要测试的攻击强度")
    parser.add_argument("--target-bpf", type=int, default=192)
    parser.add_argument("--ecc_symbols", type=int, required=True, help="该组使用的 ecc 数量 (必须准确)")
    parser.add_argument("--codec", type=str, default="libx264", choices=["libx264", "libx265"], help="使用的视频压缩编码器")
    parser.add_argument("--attack_type", type=str, default="crf", choices=["crf", "scale", "noise"])
    args = parser.parse_args()
    main(args)