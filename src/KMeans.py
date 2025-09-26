import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def init_data():
    # 使用scikit-learn库中的make_blobs函数生成一个模拟的聚类数据集
    # n_samples=200: 生成200个数据点（样本）
    # centers=3: 创建3个聚类中心（3个类别）
    # cluster_std=2.75: 每个聚类的标准差为2.75（控制聚类的紧密程度，值越大点越分散）
    # random_state=42: 随机种子，确保每次运行结果一致
    features, true_labels = make_blobs(
        n_samples=200,
        centers=3,
        cluster_std=2.75,
        random_state=42
    )
    return features, true_labels

def kmeans_basic():
    features, true_labels = init_data()

    # 使用scikit-learn库中的KMeans类来创建一个K-means聚类模型实例
    # init = "random": 指定初始化聚类中心的方法。这里使用随机初始化，即从数据集中随机选择三个点（因为n_clusters = 3）作为初始聚类中心。
    # 另一种常见的初始化方法是"k-means++"，它是智能初始化，可以加速收敛。
    #
    # n_clusters = 3: 指定要形成的聚类数量，这里设置为3，意味着算法会将数据分成3个簇。
    #
    # n_init = 100: 指定算法运行的不同初始化次数的次数。由于K - means对初始中心敏感，可能会收敛到局部最优，
    # 因此通过多次初始化并选择最佳结果（即簇内平方和最小）来缓解这个问题。这里设置为100，表示会运行100次不同的初始化，然后选择最好的一次。
    #
    # max_iter = 300: 指定单次运行中最大迭代次数。每次运行中，算法最多迭代300次，如果在此之前已经收敛（聚类中心不再变化）则会提前停止。
    #
    # random_state = 42: 随机数生成器的种子，用于确保每次运行的结果一致。这可以使得随机初始化过程可重复
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=100,
        max_iter=300,
        random_state=42
    )

    # 使用特征数据训练K-means模型
    kmeans.fit(features)

    # 打印模型评估结果
    # 惯性系数，或误差平方和(SSE)，是指所有数据点到其所属聚类中心的距离平方和
    # 值越小说明聚类效果越好（点越紧密），用于评估聚类质量和选择最佳聚类数
    print(f"惯性系数 (WCSS): {kmeans.inertia_:.2f}")
    print("聚类中心坐标:")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"聚类 {i}: ({center[0]:.2f}, {center[1]:.2f})")

def kmeans_kwargs():
    features, true_labels = init_data()

    kmeans_kwargs_params = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    # 定义一个sse list，保存每一次k值的SSE数值
    sse = []

    # K值从1到11，进行基本KMeans训练，记录每一次的SSE值，保存到sse list中
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs_params)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)

    # 将sse列表中的数据绘制成曲线图，寻找“肘部”
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()

def kmeans_silhouette_coefficient():
    features, true_labels = init_data()

    kmeans_kwargs_params = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    # 定义silhouette_coefficients list，保存每一次k值的轮廓系数
    # 轮廓系数取值范围: [-1, 1]，
    # 接近1: 聚类效果很好，簇内紧密，簇间分离
    # 接近0: 簇之间有重叠
    # 接近-1: 样本可能被分错了簇
    silhouette_coefficients = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs_params)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    # 将silhouette_coefficients列表中的数据绘制成曲线图，寻找最佳K值
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(16, 8))
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

if __name__ == '__main__':
    # 基本KMeans算法
    kmeans_basic()

    # 肘部法获取最佳K值
    kmeans_kwargs()

    # 计算KMeans轮廓系数(Silhouette Coefficient)
    kmeans_silhouette_coefficient()