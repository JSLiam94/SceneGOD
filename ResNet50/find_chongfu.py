import os
import hashlib
import shutil
from collections import defaultdict

def calculate_md5(file_path):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicate_files(folder_path):
    """查找文件夹及其子文件夹中的重复文件"""
    file_hashes = defaultdict(list)
    duplicates = []

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                file_hash = calculate_md5(file_path)
                file_hashes[file_hash].append(file_path)

    # 查找重复文件
    for hash_value, paths in file_hashes.items():
        if len(paths) > 1:
            duplicates.append(paths)

    return duplicates

def move_duplicate_files(duplicates, target_folder):
    """将重复的文件移动到目标文件夹"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for paths in duplicates:
        # 保留第一个文件，其余文件移动到目标文件夹
        for path in paths[1:]:
            try:
                # 生成目标路径，避免文件名冲突
                file_name = os.path.basename(path)
                target_path = os.path.join(target_folder, file_name)
                
                # 如果目标文件已存在，添加序号避免覆盖
                if os.path.exists(target_path):
                    base_name, ext = os.path.splitext(file_name)
                    i = 1
                    while True:
                        new_file_name = f"{base_name}_{i}{ext}"
                        target_path = os.path.join(target_folder, new_file_name)
                        if not os.path.exists(target_path):
                            break
                        i += 1
                
                shutil.move(path, target_path)
                print(f"Moved duplicate file: {path} to {target_path}")
            except Exception as e:
                print(f"Error moving file {path}: {e}")

def check_folder_overlap(folder_path):
    """检查子文件夹之间的图片是否有重叠"""
    all_files = defaultdict(list)
    folder_files = defaultdict(set)

    # 获取所有文件夹的文件信息
    for root, dirs, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        if folder_name not in folder_files:
            folder_files[folder_name] = set()
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                file_hash = calculate_md5(file_path)
                all_files[file_hash].append(file_path)
                folder_files[folder_name].add(file_hash)

    # 检查文件夹之间的重叠
    overlaps = []
    folders = list(folder_files.keys())
    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):
            folder1, folder2 = folders[i], folders[j]
            common_files = folder_files[folder1].intersection(folder_files[folder2])
            if common_files:
                overlaps.append((folder1, folder2, len(common_files)))

    return overlaps

def main():
    folder_path = "E:\RefCOD\img"  # 替换为你的图片文件夹路径
    duplicate_folder = os.path.join(folder_path, "duplicates")

    # 查找重复文件
    duplicates = find_duplicate_files(folder_path)
    if duplicates:
        print("发现重复文件，开始移动...")
        move_duplicate_files(duplicates, duplicate_folder)
    else:
        print("未发现重复文件")

    # 检查子文件夹之间的重叠
    overlaps = check_folder_overlap(folder_path)
    if overlaps:
        print("\n发现子文件夹之间有重叠的图片:")
        for folder1, folder2, count in overlaps:
            print(f"文件夹 '{folder1}' 和 '{folder2}' 之间有 {count} 个重叠的图片")
    else:
        print("\n未发现子文件夹之间有重叠的图片")

if __name__ == "__main__":
    main()