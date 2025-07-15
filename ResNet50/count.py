import os

def count_images_in_folder(folder_path):
    """统计文件夹及其子文件夹中的图片数量"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    total_images = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1

    return total_images

def main():
    folder_path = "E:\TrainDataset\Imgs"  # 替换为你的图片文件夹路径

    total_images = count_images_in_folder(folder_path)
    print(f"文件夹 '{folder_path}' 及其子文件夹中共有 {total_images} 张图片")

if __name__ == "__main__":
    main()