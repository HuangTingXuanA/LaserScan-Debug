import cv2
import numpy as np

# 读取灰度图
img = cv2.imread("circle2.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("无法读取图像 circle.png")

# 阈值分割（灰度 >= 200）
_, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 转为彩色方便画图
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

centers = []

for cnt in contours:
    if len(cnt) < 5:
        continue  # 排除无效轮廓
    
    # 检查半径是否 >= 30
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    if radius < 30:
        continue  # 半径太小，跳过

    # 计算质心
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 亚像素精度
    corner = np.array([[[cx, cy]]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    subpix = cv2.cornerSubPix(img.astype(np.float32), corner, (5,5), (-1,-1), criteria)

    sx, sy = subpix[0,0]
    centers.append((sx, sy))

    # 可视化圆心
    cv2.circle(vis, (int(round(sx)), int(round(sy))), 3, (0,0,255), -1)
    cv2.putText(vis, f"({sx:.2f},{sy:.2f})", (int(sx)+5, int(sy)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

# 显示结果
cv2.imshow("Centers", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()