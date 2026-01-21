import torch
import os

def inspect_structure(file_path):
    """
    智能加载：自动寻找被包裹在 'net', 'state_dict' 等键下的参数
    """
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None
        
    try:
        # map_location='cpu' 保证在任何机器上都能跑
        ckpt = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"无法加载文件: {e}")
        return None

    if isinstance(ckpt, dict):
        # 优先查找常见的包裹键
        for key in ['net', 'model', 'state_dict', 'model_state_dict']:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"-> 在文件 '{os.path.basename(file_path)}' 中自动提取键: '{key}'")
                return ckpt[key]
    
    return ckpt

def print_list_limited(title, items, limit=50):
    """
    辅助函数：打印列表，超过限制则省略
    """
    count = len(items)
    print(f"\n{title} (共 {count} 个):")
    
    if count == 0:
        print("    (无)")
        return

    for i, item in enumerate(items):
        if i >= limit:
            print(f"    ... 以及其他 {count - limit} 个 (已隐藏)")
            break
            
        if isinstance(item, tuple):
            # 格式: 参数名 (差异值)
            print(f"    {item[0]} \t[平均差异: {item[1]:.6f}]")
        else:
            # 格式: 参数名
            print(f"    {item}")

def compare_weights_names(path_a, path_b):
    print(f"{'='*10} 正在加载文件 {'='*10}")
    sd_a = inspect_structure(path_a)
    sd_b = inspect_structure(path_b)

    if sd_a is None or sd_b is None:
        return

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    intersection = keys_a & keys_b

    matches = []
    diff_values = []
    diff_shapes = []
    
    print(f"\n{'='*10} 正在逐层比对 {'='*10}")
    
    for key in intersection:
        val_a = sd_a[key]
        val_b = sd_b[key]
        
        # 忽略非 Tensor 数据
        if not torch.is_tensor(val_a) or not torch.is_tensor(val_b):
            continue

        # 1. 形状不同
        if val_a.shape != val_b.shape:
            diff_shapes.append(f"{key} (A:{val_a.shape} vs B:{val_b.shape})")
            continue
            
        # 2. 数值相同
        if torch.equal(val_a, val_b):
            matches.append(key)
        
        # 3. 数值不同 (计算差异)
        else:
            # === 修复点在这里 ===
            # 先转为 .float() 再计算平均值，防止 Long 类型报错
            diff = (val_a.float() - val_b.float()).abs().mean().item()
            diff_values.append((key, diff))

    # 排序
    matches.sort()
    diff_values.sort(key=lambda x: x[0])
    diff_shapes.sort()
    only_a = sorted(list(keys_a - keys_b))
    only_b = sorted(list(keys_b - keys_a))

    # 打印限制 (防止刷屏，想看全部可以将 100 改为 9999)
    print_limit = 100 

    print_list_limited("[1] 完全相同的参数 (Pre-trained/Frozen)", matches, limit=print_limit)
    print_list_limited("[2] 数值不同的参数 (Trained/Updated)", diff_values, limit=print_limit)
    print_list_limited("[3] 形状不匹配", diff_shapes, limit=print_limit)
    print_list_limited("[4] 仅在文件 A 中", only_a, limit=print_limit)
    print_list_limited("[5] 仅在文件 B 中", only_b, limit=print_limit)

    print(f"\n{'='*30}\n比对完成。")

if __name__ == "__main__":
    # 你的文件路径
    file_a = r"A:\AI\Project\OSTrack\output\checkpoints\train\ostrack\OSTrack_mem_55_roi8_hy1.0_pee_V1.9\OSTrack_ep0100.pth.tar"
    # file_a = r"A:\AI\Project\OSTrack\pretrained\OSTrack_pretrained_old.tar"
    file_b = r"A:\AI\Project\OSTrack\pretrained\OSTrack_pretrained.tar"

    compare_weights_names(file_a, file_b)