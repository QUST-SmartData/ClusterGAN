'''在本视频中，您学习了如何使用 k-means 惯性图为数据集选择大量聚类。您将获得一个数组，其中包含谷物样本的测量值（例如面积、周长、长度和其他几个测量值）。在这种情况下，什么是好的集群数量？'''

'''代码会绘制簇的数量 k 与惯性（inertia）之间的关系图，这有助于找到“肘点”。在图中，惯性值开始减缓下降的位置即为“肘点”，通常被认为是合适的簇数量的选择。

'''




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
samples = np.loadtxt("./runs/my_dataset/20quan-c600s-1e-4-210epoch_z64_wass_bs4_test_run/encoded_latent_vectors.csv", delimiter=",", skiprows=1)
ks = range(1, 8)
inertias = []

for k in ks:
    # 创建一个具有 k 个簇的 KMeans 实例：model
    model = KMeans(n_clusters=k)
    
    # 将模型拟合到样本数据
    model.fit(samples)
    
    # 将模型的 inertia 值追加到 inertias 列表中
    inertias.append(model.inertia_)
    
# 绘制 ks 与 inertias 的关系图
plt.plot(ks, inertias, '-o')
plt.xlabel(' k')
plt.ylabel('(inertia)')
plt.xticks(ks)
plt.savefig('./2000inertia.png')
