from PIL import Image
import imageio
import os
import re
import numpy as np
import pdb

def sort_key(file_name):
    # 使用正则表达式提取文件名中的数字
    numbers = re.findall(r'\d+', file_name)
    # 将提取到的数字转换为整数，并返回一个元组，方便排序
    return tuple(map(int, numbers)) if numbers else (0,)


scene_path = '/media/sun/z/argoverse2/sensor/train/2451c219-3002-3b2e-8fa9-2b7fea168b3b/sensors/cameras/ring_front_center/'
# 获取文件夹中所有文件的文件名
file_names = os.listdir(scene_path)
file_names_sorted = sorted(file_names, key=sort_key)


def resize_image(image_path):
    with Image.open(image_path) as img:
        new_width = (img.width + 15) // 16 * 16
        new_height = (img.height + 15) // 16 * 16
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img.convert("RGBA")
def save_as_video(image_list, mp4_output_path, scale=None):
    mp4_output_path = mp4_output_path.replace('.gif', '.mp4')
    # images = [Image.fromarray(imageio.imread(img_path)).convert("RGBA") for img_path in image_list]
    images = [resize_image(img_path) for img_path in image_list]
    if scale is not None:
        w, h = images[0].size
        images = [img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS) for img in images]
    # images = [Image.new('RGBA', images[0].size, (255, 255, 255, 255))] + images

    try:
        imageio.mimsave(mp4_output_path, images, format='MP4', fps=30)
    except ValueError:  # in case the shapes are not the same, have to manually adjust
        resized_images = [img.resize(images[0].size, Image.Resampling.LANCZOS) for img in images]
        print('Size not all the same, manually adjust...')
        imageio.mimsave(mp4_output_path, resized_images, format='MP4', fps=10)
    print("mp4 saved to : ", mp4_output_path)


def save_as_video1(image_list, mp4_output_path, fps=30):
    images = []
    for img_path in image_list:
        img = Image.open(img_path).convert("RGBA")
        img_np = np.array(img)
        images.append(img_np)

    # 使用imageio.get_writer显式指定编码器
    with imageio.get_writer(mp4_output_path, fps=fps, codec='libx264') as writer:
        for image in images:
            writer.append_data(image)

image_list = [scene_path + frame_timestep for frame_timestep in file_names_sorted]
gif_output_path = '/home/sun/Bev/BeMapNet/zknecht/vis_unmerge' + '/2451c219-3002-3b2e-8fa9-2b7fea168b3b.gif'
save_as_video1(image_list, gif_output_path)