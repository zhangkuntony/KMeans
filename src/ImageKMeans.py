import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def image_kmeans():
    image = load_image()

    # 将三维图像数组转换为二维数组，每个像素点变成一行数据。
    # -1 表示自动计算该维度大小，保持总元素数不变
    # 3 表示保留RGB三个通道值
    # 假设图像像素为800*600，转换之后为 800像素高度 × 600像素宽度 × 3个颜色通道(RGB)
    pixel_vals = image.reshape((-1, 3))

    # 将数据类型从整数（通常是uint8，0-255范围）转换为32位浮点数。
    # K-means算法要求：聚类算法通常需要浮点数输入
    # 数值稳定性：浮点数计算更精确，避免整数运算的截断误差
    # 距离计算：聚类基于距离度量，浮点数更适合距离计算
    pixel_vals = np.float32(pixel_vals)

    kmeans = KMeans(
        init='random',
        n_clusters=10,
    )

    kmeans.fit(pixel_vals)

    # 将每个像素点替换为其所属聚类的中心颜色值，实现图像分割
    segmented_data = kmeans.cluster_centers_[kmeans.predict(pixel_vals)]
    # 将浮点数值转换回8位无符号整数（0-255范围），这是图像的标准格式
    segmented_data = np.uint8(segmented_data)
    # 重新设置segmented_data数据的尺寸和原始图片尺寸一致
    segmented_image = segmented_data.reshape(image.shape)

    plt.imshow(segmented_image)
    plt.show()

def load_image():
    # 加载图片./images/dog-cat.jpeg
    image = cv2.imread("./images/dog-cat.jpeg")

    # 颜色空间转换
    # 重要: OpenCV默认使用BGR颜色格式，但Matplotlib使用RGB格式
    # 这一行将图像从BGR转换为RGB，确保显示时颜色正确
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 设置绘图样式为默认
    plt.style.use('default')
    # 隐藏X，Y轴的刻度标签
    _ = plt.xticks(); _ = plt.yticks()
    # 显示图像
    plt.imshow(image)
    plt.show()

    return image

if __name__ == "__main__":
    image_kmeans()