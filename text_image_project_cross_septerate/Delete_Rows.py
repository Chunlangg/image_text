import os
from PIL import Image
import pandas as pd

# 输入文件路径和输出文件路径

output_csv = 'output_final_data.csv'

def is_image_valid(image_path):
    """check if the iamge can be open"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # 验证图片文件是否完整
        return True
    except (IOError, FileNotFoundError):
        return False

# 读取CSV文件
df = pd.read_csv('/Users/chunlan/Research/simple_project_newest /code/image_text/final_data.csv')

# 假设有一列 'image_path' 存储图片路径
# 通过is_image_valid函数过滤出有效的图片路径行
df_valid = df[df['media_path'].apply(lambda x: os.path.exists(x) and is_image_valid(x))]

# 将有效的行保存到新的CSV文件中
df_valid.to_csv(output_csv, index=False)

print(f"处理完成！无效的图片路径已删除，并保存到 {output_csv}")
