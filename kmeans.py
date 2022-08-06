import numpy as np
import random
import matplotlib.pyplot as plt

def distance(point1,point2):
    return np.sqrt(np.sum(point1-point2)**2)

def k_means(data,k,max_iter=10000):
    centers = {} #初始聚类中心
    #初始化，随机选K个样本作为初始聚类中心
    #random.sample():随机不重复抽取K个值
    n_data = data.shape[0] #样本个数
    for idx,i in enumerate(random.sample(range(n_data),k)):
        #idx取值范围为[0，k-1]，代表第几个聚类中心
        centers[idx] = data[i]
    #开始迭代
    for i in range(max_iter): #迭代次数
        print("epoch{}".format(i+1))
        clusters={} #聚类结果，聚类中心的索引index ->样本集合
        for j in range(k): #初始化为空列表
            clusters[j] = []

        for sample in data: #遍历每个样本
            distances = []
            for c in centers:
                distances.append(distance(sample,centers[c]))
            idx = np.argmin(distances) #最小距离的索引
            clusters[idx].append(sample)
        
        pre_centers = centers.copy()

        for c in clusters.keys():
            centers[c] = np.mean(clusters[c],axis=0)
        
        is_convergent=True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:  
            # 如果新旧聚类中心不变，则迭代停止
            break
    return centers, clusters

def predict(p_data,centers):
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances = [distance(p_data, centers[c]) for c in centers]  
    return np.argmin(distances)

if __name__=='__main__':
    x = np.random.randint(0,high=10,size=(100,2))
    print(x)
    centers,clusters = k_means(x,4)
    for center in centers:
        plt.scatter(centers[center][0],centers[center][1],marker='*',s=350)

    colors=['r','b','y','m','c','g']
    for c in clusters:
        for point in clusters[c]:
            plt.scatter(point[0],point[1],c =colors[c])
    plt.show()
