# 文件名: stego_ecc.py
import reedsolo

# 16 个校验字节
ECC_SYMBOLS = 16
rs = reedsolo.RSCodec(ECC_SYMBOLS)

def encode_message_rs_bits(text: str) -> str:
    """发件端：将文本直接转为带有 RS 校验的 0/1 字符串"""
    data_bytes = text.encode('utf-8')
    encoded_bytes = rs.encode(data_bytes)
    
    # 直接转为 0/1 字符串，传给底层的 stego_processor
    bit_str = "".join([bin(b)[2:].zfill(8) for b in encoded_bytes])
    
    print(f"\n[ECC 引擎] 护甲穿戴完毕！长度: {len(data_bytes)}B -> {len(encoded_bytes)}B")
    print(f"[ECC 引擎] 底层物理载荷(msg-len)需设为: {len(bit_str)} bits\n")
    return bit_str

def decode_message_rs_bits(bits) -> str:
    """收件端：拦截底层的 0/1 数组，直接进行纠错解码"""
    # 1. 组装回字节流 (支持 numpy array 或 list)
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_chunk = bits[i:i+8]
        if len(byte_chunk) == 8:
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
            
    # 2. RS 纠错解码
    try:
        decoded_bytes, _, err_pos = rs.decode(byte_array)
        recovered_text = decoded_bytes.decode('utf-8')
        print(f"\n[ECC 引擎] ✅ 触发神级纠错！成功修复了 {len(err_pos)} 个错误字节！")
        return recovered_text
    except reedsolo.ReedSolomonError:
        print("\n[ECC 引擎] ❌ 纠错失败：误码率超出 RS 极限。")
        # 降级返回（方便观察残骸）
        return "".join([chr(b) if 32 <= b <= 126 else '?' for b in byte_array])
    except Exception as e:
        print(f"\n[ECC 引擎] ❌ 解析异常: {e}")
        return "[解析失败: 编码异常]"