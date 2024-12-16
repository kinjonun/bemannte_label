import cv2
import numpy as np

# 创建一个空白图像
mask = np.zeros((512, 512, 3), dtype=np.uint8)

# 定义多边形的顶点坐标
coords = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)

# 使用 cv2.polylines 绘制多边形折线
cv2.polylines(mask, [coords], isClosed=False, color=(0, 255, 0), thickness=1)

# 显示图像
cv2.imshow('Polylines', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
