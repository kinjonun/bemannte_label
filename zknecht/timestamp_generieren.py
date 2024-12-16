import os
import re
import pdb
def sort_key(file_name):
    # 使用正则表达式提取文件名中的数字
    numbers = re.findall(r'\d+', file_name)
    # 将提取到的数字转换为整数，并返回一个元组，方便排序
    return tuple(map(int, numbers)) if numbers else (0,)

# 指定目标文件夹路径
# folder_path = '/home/sun/Bev/BeMapNet/data/argoverse2/customer_'
folder_path = '/media/sun/z/argoverse2/sensor/test'
scene_path = '/media/sun/z/argoverse2/sensor/train/22dcf96c-ef5e-376b-9db5-dc9f91040f5e/sensors/cameras/ring_front_center'
# 获取文件夹中所有文件的文件名
file_names = os.listdir(scene_path)
file_names_sorted = sorted(file_names, key=sort_key)
# pdb.set_trace()
# 指定输出txt文件的路径
output_file = '/home/sun/Bev/BeMapNet/zknecht/vis_unmerge/22dcf96c-ef5e-376b-9db5-dc9f91040f5e.txt'

# 将文件名写入到txt文件中
with open(output_file, 'w', encoding='utf-8') as f:
    for file_name in file_names_sorted:
        name_without_extension = os.path.splitext(file_name)[0]
        f.write(name_without_extension + '\n')

print(f"文件名已成功写入到 {output_file}")

