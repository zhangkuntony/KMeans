# KMeans 聚类算法实现与应用

这个项目展示了K-Means聚类算法在不同场景下的实现和应用，包括基础聚类分析、图像分割和用户行为分析。

## 项目结构

```
KMeans/
├── src/                   # 源代码目录
│   ├── KMeans.py          # 基础K-Means聚类实现
│   ├── ImageKMeans.py     # 图像分割应用
│   └── UserBehaviorKMeans.py  # 用户行为分析应用
├── data/                  # 数据文件
│   └── UserBehavior-15k.csv  # 用户行为数据集
└── images/                # 图像文件
    └── dog-cat.jpeg       # 用于图像分割的示例图片
```

## 功能模块

### 1. 基础聚类分析 (src/KMeans.py)

实现了K-Means聚类算法的基本功能，包括：

- 生成模拟聚类数据集
- 基本K-Means聚类实现
- 使用肘部法则(Elbow Method)确定最佳聚类数
- 使用轮廓系数(Silhouette Coefficient)评估聚类质量

主要函数：
- `init_data()`: 生成模拟聚类数据
- `kmeans_basic()`: 基本K-Means聚类实现
- `kmeans_kwargs()`: 使用肘部法则确定最佳K值
- `kmeans_silhouette_coefficient()`: 计算轮廓系数评估聚类质量

### 2. 图像分割 (src/ImageKMeans.py)

使用K-Means算法进行图像分割，将图像中的像素点按颜色特征进行聚类，实现图像的颜色量化。

主要函数：
- `load_image()`: 加载并显示原始图像
- `image_kmeans()`: 对图像进行K-Means聚类分割

### 3. 用户行为分析 (src/UserBehaviorKMeans.py)

分析用户行为数据，使用K-Means算法对用户进行分群，发现不同的用户行为模式。

主要函数：
- `load_user_behavior_data()`: 加载用户行为数据
- `calc_user_action_count()`: 计算用户行为特征
- `user_behavior_kmeans()`: 对用户行为进行聚类分析

## 数据集

### UserBehavior-15k.csv

用户行为数据集，包含以下字段：
- user: 用户ID
- item: 商品ID
- category: 商品类别
- action: 用户行为类型
- ts: 时间戳

## 环境要求

- Python 3.6+
- 依赖库：
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - opencv-python (cv2)

## 使用方法

### 基础聚类分析

```bash
python src/KMeans.py
```

### 图像分割

```bash
python src/ImageKMeans.py
```

### 用户行为分析

```bash
python src/UserBehaviorKMeans.py
```

## 算法说明

### K-Means 聚类算法

K-Means是一种常用的聚类算法，其基本步骤如下：

1. 随机选择K个点作为初始聚类中心
2. 将每个数据点分配到最近的聚类中心
3. 重新计算每个聚类的中心点
4. 重复步骤2和3，直到聚类中心不再变化或达到最大迭代次数

### 评估指标

- **惯性系数(WCSS)**: 所有数据点到其所属聚类中心的距离平方和，值越小表示聚类效果越好
- **轮廓系数(Silhouette Coefficient)**: 衡量聚类的紧密度和分离度，取值范围[-1,1]，越接近1表示聚类效果越好

## 应用场景

- **数据分析**: 发现数据中的自然分组和模式
- **图像处理**: 图像分割、颜色量化、特征提取
- **用户画像**: 用户分群、行为分析、个性化推荐

## 参考资料

- [K-Means聚类算法原理](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [轮廓系数评估方法](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [肘部法则确定最佳K值](https://en.wikipedia.org/wiki/Elbow_method_(clustering))