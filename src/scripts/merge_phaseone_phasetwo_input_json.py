"""
合并 phaseone 和 phasetwo 的 input.json 文件

功能说明：
1. 读取 phaseone/input.json
2. 读取 phasetwo/input.json  
3. 合并两个 JSON 数组
4. 保存到 input/input.json

目录结构：
- 脚本位置: scripts/merge_phaseone_phasetwo_input_json.py
- phaseone数据: phaseone/input.json
- phasetwo数据: phasetwo/input.json
- 输出位置: input/input.json

使用方法：
python merge_phaseone_phasetwo_input_json.py
"""

import json
import os

def merge_input_json():
    """合并 phaseone 和 phasetwo 的 input.json 文件"""
    
    # 获取脚本所在目录（scripts目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(script_dir)
    
    # 定义文件路径
    phaseone_file = os.path.join(project_root, 'phaseone', 'input.json')
    phasetwo_file = os.path.join(project_root, 'phasetwo', 'input.json')
    
    # 确保 input 目录存在
    input_dir = os.path.join(project_root, 'input')
    os.makedirs(input_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(input_dir, 'input.json')
    
    print("开始合并 input.json 文件...")
    print(f"phaseone 源文件: {phaseone_file}")
    print(f"phasetwo 源文件: {phasetwo_file}")
    print(f"输出文件: {output_file}")
    
    # 检查文件是否存在
    phaseone_exists = os.path.exists(phaseone_file)
    phasetwo_exists = os.path.exists(phasetwo_file)
    
    if not phaseone_exists and not phasetwo_exists:
        print("错误: 未找到任何 input.json 文件")
        return False
    
    merged_data = []
    
    # 读取 phaseone 数据
    if phaseone_exists:
        try:
            with open(phaseone_file, 'r', encoding='utf-8') as f:
                phaseone_data = json.load(f)
                merged_data.extend(phaseone_data)
                print(f"✅ 成功读取 phaseone 数据: {len(phaseone_data)} 条记录")
        except json.JSONDecodeError as e:
            print(f"❌ 错误: phaseone input.json 格式错误 - {e}")
            return False
        except Exception as e:
            print(f"❌ 错误: 读取 phaseone input.json 失败 - {e}")
            return False
    else:
        print("⚠️  警告: 未找到 phaseone/input.json 文件")
    
    # 读取 phasetwo 数据
    if phasetwo_exists:
        try:
            with open(phasetwo_file, 'r', encoding='utf-8') as f:
                phasetwo_data = json.load(f)
                merged_data.extend(phasetwo_data)
                print(f"✅ 成功读取 phasetwo 数据: {len(phasetwo_data)} 条记录")
        except json.JSONDecodeError as e:
            print(f"❌ 错误: phasetwo input.json 格式错误 - {e}")
            return False
        except Exception as e:
            print(f"❌ 错误: 读取 phasetwo input.json 失败 - {e}")
            return False
    else:
        print("⚠️  警告: 未找到 phasetwo/input.json 文件")
    
    if not merged_data:
        print("❌ 错误: 没有找到任何有效数据")
        return False
    
    # 保存合并后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 成功保存合并文件到: {output_file}")
        print(f"📊 合并后总计: {len(merged_data)} 条记录")
    except Exception as e:
        print(f"❌ 错误: 保存合并文件失败 - {e}")
        return False
    
    print("\n🎉 input.json 文件合并完成！")
    return True

if __name__ == "__main__":
    success = merge_input_json()
    if not success:
        print("\n💥 合并失败!")
        exit(1)