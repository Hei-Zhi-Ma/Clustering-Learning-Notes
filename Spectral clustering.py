"""
spectral_clustering.py
谱聚类的简洁且注释完善的实现（用于学习目的）
支持功能：
 - 非归一化拉普拉斯矩阵和归一化拉普拉斯矩阵
 - 稀疏k近邻亲和矩阵构建
 - 使用第二小特征向量（Fiedler向量）进行二分类划分
 - 谱嵌入（取前k个特征向量）+ k-means实现多分类
 - 支持稀疏特征值求解器（eigsh）以处理大规模数据

使用方法：
  from spectral_clustering import spectral_clustering, affinity_matrix
  labels = spectral_clustering(X, n_clusters=2, sigma=0.5, laplacian='sym', k_nn=10)
"""

# 导入必要的库
import numpy as np
from scipy.spatial.distance import cdist  # 计算距离矩阵
from scipy.linalg import eigh  # 用于对称稠密矩阵的特征值分解
from scipy.sparse.linalg import eigsh  # 用于大规模稀疏矩阵的特征值分解
from sklearn.cluster import KMeans  # k-means聚类算法
from sklearn.neighbors import NearestNeighbors  # k近邻搜索
from scipy import sparse  # 稀疏矩阵操作

# -----------------------------
# 工具函数：亲和矩阵构建
# 核心作用：将原始数据转换为表示样本间相似度的矩阵
# -----------------------------
def gaussian_affinity(X, sigma=1.0, mode='full', k=10, sym=True):
    """
    从数据X构建亲和矩阵S（相似度矩阵）

    参数说明
    ----------
    X : 二维数组 (样本数量, 特征维度)
        输入的原始数据
    sigma : 浮点数, 高斯核的带宽参数
        控制相似度的衰减速度，sigma越大，相似度衰减越慢
    mode : 'full' 或 'knn' 
        构建亲和矩阵的模式：
        - 'full'：全连接矩阵（稠密矩阵）
        - 'knn'：k近邻图（稀疏矩阵）
    k : 整数, 当mode='knn'时的近邻数量
    sym : 布尔值, 若为True，通过(S+S.T)/2确保稀疏矩阵的对称性

    返回值
    -------
    S : 二维数组或scipy稀疏矩阵(csr格式)
        构建好的亲和矩阵
    """
    n = X.shape[0]  # 获取样本数量
    if mode == 'full':
        # 全连接高斯核矩阵（时间复杂度O(n²)），适用于样本量较小的场景
        # 计算所有样本对之间的平方欧氏距离
        dists = cdist(X, X, metric='sqeuclidean')
        # 高斯核公式：S_ij = exp(-||x_i - x_j||² / (2σ²))
        S = np.exp(-dists / (2.0 * sigma * sigma))
        # 对角线元素设为0（样本与自身的相似度不考虑）
        np.fill_diagonal(S, 0.0)
        return S
    elif mode == 'knn':
        # 稀疏k近邻图：适用于大规模数据，节省计算量和内存
        # 构建k+1个近邻的搜索器（+1是因为会包含样本自身）
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        # 获取每个样本的k+1个近邻的距离和索引
        distances, indices = nbrs.kneighbors(X)
        # 构建稀疏矩阵的行索引：每个样本重复k次（对应k个近邻）
        rows = np.repeat(np.arange(n), k)
        # 构建列索引：跳过自身索引（indices[:,0]是样本自身），取后k个近邻
        cols = indices[:, 1:(k+1)].flatten()
        # 计算近邻样本对的高斯相似度
        vals = np.exp(- (distances[:, 1:(k+1)].flatten()**2) / (2.0 * sigma * sigma))
        # 构建稀疏矩阵（csr格式：按行压缩）
        S = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        # 确保矩阵对称性
        if sym:
            S = 0.5 * (S + S.T)
        # 对角线元素设为0
        S.setdiag(0.0)
        # 移除零元素以优化存储
        S.eliminate_zeros()
        return S
    else:
        raise ValueError("mode参数必须是'full'或'knn'")

# -----------------------------
# 拉普拉斯矩阵构造函数
# 核心作用：从亲和矩阵转换为拉普拉斯矩阵，是谱聚类的核心步骤
# -----------------------------
def degree_matrix_from_S(S):
    """
    从亲和矩阵S计算度矩阵D（对角矩阵）

    度矩阵的定义：对角线元素D_ii = 亲和矩阵S第i行的和（样本i的连接强度总和）
    返回：稠密对角数组或稀疏对角矩阵
    """
    if sparse.issparse(S):
        # 稀疏矩阵处理：计算每行的和并转换为一维数组
        degs = np.array(S.sum(axis=1)).flatten()
        return sparse.diags(degs)  # 返回稀疏对角矩阵
    else:
        # 稠密矩阵处理：计算每行的和并构建对角矩阵
        degs = np.sum(S, axis=1)
        return np.diag(degs)  # 返回稠密对角矩阵

def laplacian(S, kind='unnormalized'):
    """
    计算拉普拉斯矩阵

    参数：
        kind: 拉普拉斯矩阵类型
            - 'unnormalized'：非归一化拉普拉斯矩阵 (L = D - S)
            - 'sym'：对称归一化拉普拉斯矩阵 (L_sym = I - D^(-1/2) S D^(-1/2))
            - 'rw'：随机游走归一化拉普拉斯矩阵 (L_rw = I - D^(-1) S)
    返回：
        L: 拉普拉斯矩阵（稠密或稀疏格式，与输入S一致）
    """
    if sparse.issparse(S):
        # 稀疏矩阵处理路径
        # 计算每个样本的度（每行求和）
        degs = np.array(S.sum(axis=1)).flatten()
        if kind == 'unnormalized':
            # 非归一化拉普拉斯矩阵：度矩阵减亲和矩阵
            D = sparse.diags(degs)
            return D - S
        elif kind == 'sym':
            # 对称归一化拉普拉斯矩阵
            # 避免度为0时的除法错误，添加微小值
            degs_eps = degs.copy()
            degs_eps[degs_eps == 0] = 1e-12
            # 构建D的逆平方根矩阵
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degs_eps))
            # 单位矩阵
            I = sparse.identity(S.shape[0], format='csr')
            return I - D_inv_sqrt.dot(S).dot(D_inv_sqrt)
        elif kind == 'rw':
            # 随机游走归一化拉普拉斯矩阵
            degs_eps = degs.copy()
            degs_eps[degs_eps == 0] = 1e-12
            # 构建D的逆矩阵
            D_inv = sparse.diags(1.0 / degs_eps)
            I = sparse.identity(S.shape[0], format='csr')
            return I - D_inv.dot(S)
        else:
            raise ValueError("kind参数无效，必须是'unnormalized'、'sym'或'rw'")
    else:
        # 稠密矩阵处理路径
        degs = np.sum(S, axis=1)
        if kind == 'unnormalized':
            D = np.diag(degs)
            return D - S
        elif kind == 'sym':
            degs_eps = degs.copy()
            degs_eps[degs_eps == 0] = 1e-12
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degs_eps))
            I = np.eye(S.shape[0])
            return I - D_inv_sqrt.dot(S).dot(D_inv_sqrt)
        elif kind == 'rw':
            degs_eps = degs.copy()
            degs_eps[degs_eps == 0] = 1e-12
            D_inv = np.diag(1.0 / degs_eps)
            I = np.eye(S.shape[0])
            return I - D_inv.dot(S)
        else:
            raise ValueError("kind参数无效，必须是'unnormalized'、'sym'或'rw'")

# -----------------------------
# 特征值求解器包装器
# 核心作用：统一处理稠密/稀疏矩阵的特征值分解，获取最小的k个特征值和特征向量
# -----------------------------
def smallest_k_eig(L, k=2, sparse_tol=1000):
    """
    计算对称矩阵L的最小k个特征值和对应的特征向量

    策略：如果矩阵是稀疏的或样本量较大，使用稀疏特征值求解器eigsh，否则使用稠密求解器eigh
    """
    n = L.shape[0]
    # 启发式判断：稀疏矩阵或样本数超过阈值时使用稀疏求解器
    if sparse.issparse(L) or n > sparse_tol:
        # eigsh默认求解最大的特征值，通过which='SM'指定求解最小模的特征值
        try:
            vals, vecs = eigsh(L, k=k, which='SM')
        except Exception as e:
            # 异常处理：将稀疏矩阵转为稠密矩阵后求解（适用于求解失败的情况）
            vals_all, vecs_all = eigh(L.toarray() if sparse.issparse(L) else L)
            vals = vals_all[:k]  # 取前k个最小的特征值
            vecs = vecs_all[:, :k]  # 对应特征向量
    else:
        # 稠密矩阵直接求解，eigh默认返回升序排列的特征值
        vals_all, vecs_all = eigh(L)
        vals = vals_all[:k]
        vecs = vecs_all[:, :k]
    # 确保特征值按升序排列（处理求解器返回顺序可能不一致的情况）
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

# -----------------------------
# 谱聚类主函数
# 整合所有步骤：亲和矩阵→拉普拉斯矩阵→特征分解→聚类
# -----------------------------
def spectral_clustering(X, n_clusters=2, sigma=1.0, affinity_mode='knn', k=10,
                        laplacian_kind='sym', eigen_k=None, use_sign=False,
                        random_state=0):
    """
    高层谱聚类函数，实现完整的谱聚类流程

    参数说明
    ----------
    X : 二维数组, 形状 (样本数量, 特征维度)
        输入的原始数据
    n_clusters : 整数, 目标聚类数量
    sigma : 浮点数, 高斯核的带宽参数
    affinity_mode : 'full' 或 'knn'，亲和矩阵构建模式
    k : 整数, 当affinity_mode='knn'时的近邻数量
    laplacian_kind : 拉普拉斯矩阵类型 ('unnormalized', 'sym', 'rw')
    eigen_k : 整数, 要计算的特征向量数量（默认等于n_clusters）
    use_sign : 布尔值, 若为True且n_clusters==2时，使用Fiedler向量的符号进行二分类
              （不使用k-means）；若为False，使用谱嵌入+k-means
    random_state : 整数, 随机种子（保证k-means结果可复现）

    返回值
    -------
    labels : 一维数组 (样本数量,)
        每个样本的聚类标签
    info : 字典
        包含中间结果的字典（亲和矩阵S、拉普拉斯矩阵L、特征值eigvals、特征向量eigvecs）
    """
    # 若未指定特征向量数量，默认等于聚类数量
    if eigen_k is None:
        eigen_k = n_clusters

    # 步骤1：构建亲和矩阵（样本间相似度）
    S = gaussian_affinity(X, sigma=sigma, mode=affinity_mode, k=k, sym=True)

    # 步骤2：构建拉普拉斯矩阵
    L = laplacian(S, kind=laplacian_kind)

    # 步骤3：特征分解，获取最小的eigen_k个特征值和特征向量
    eigvals, eigvecs = smallest_k_eig(L, k=eigen_k)

    # 存储中间结果
    info = {'S': S, 'L': L, 'eigvals': eigvals, 'eigvecs': eigvecs}

    # 特殊情况：二分类时使用Fiedler向量的符号进行划分
    if use_sign and n_clusters == 2:
        # 确保至少有2个特征向量（Fiedler向量是第二小的特征向量）
        if eigvecs.shape[1] < 2:
            # 重新计算前2个特征向量
            eigvals, eigvecs = smallest_k_eig(L, k=2)
            info['eigvals'] = eigvals
            info['eigvecs'] = eigvecs
        # Fiedler向量：第二小的特征向量（对应聚类的划分信息）
        fiedler = eigvecs[:, 1]
        # 根据Fiedler向量的符号分配标签（正为1，负为0）
        labels = (fiedler > 0).astype(int)
        return labels, info

    # 通用多分类情况：谱嵌入 + k-means
    # 确保特征向量数量不少于聚类数量
    if eigvecs.shape[1] < n_clusters:
        eigvals, eigvecs = smallest_k_eig(L, k=n_clusters)
        info['eigvals'] = eigvals
        info['eigvecs'] = eigvecs

    # 步骤4：谱嵌入：取前n_clusters个特征向量作为嵌入空间
    U = eigvecs[:, :n_clusters]  # 形状 (样本数量, 聚类数量)

    # 可选步骤：对嵌入向量进行行归一化（对称拉普拉斯矩阵常用技巧，提升聚类效果）
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    # 避免除以零的情况
    row_norms[row_norms == 0] = 1e-12
    U_normalized = U / row_norms

    # 步骤5：在嵌入空间上运行k-means聚类
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(U_normalized)

    return labels, info

# -----------------------------
# 演示函数：用于测试和学习，展示关键中间步骤的结果
# -----------------------------
def demo_small_example():
    """
    小型演示案例：使用手动定义的3x3亲和矩阵展示谱聚类的关键步骤
    """
    # 手动定义的3x3亲和矩阵（模拟3个样本的相似度）
    S = np.array([[0.0, 0.9, 0.1],
                  [0.9, 0.0, 0.2],
                  [0.1, 0.2, 0.0]])
    # 计算度矩阵
    D = np.diag(S.sum(axis=1))
    # 计算非归一化拉普拉斯矩阵
    L = D - S

    # 打印关键中间结果
    print("亲和矩阵 S=\n", S)
    print("度矩阵 D=\n", D)
    print("非归一化拉普拉斯矩阵 L=\n", L)

    # 特征分解
    eigvals, eigvecs = eigh(L)
    print("特征值（升序）=\n", eigvals)
    print("特征向量（列对应特征值）=\n", eigvecs)
    print("Fiedler向量（第二小特征向量）≈\n", eigvecs[:, 1])

    # 基于Fiedler向量符号的二分类
    labels_sign = (eigvecs[:, 1] > 0).astype(int)
    print("基于Fiedler向量符号的标签:", labels_sign)

    # 演示谱嵌入+k-means的过程
    U = eigvecs[:, 1:2]  # 使用Fiedler向量作为嵌入特征
    # 行归一化
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    Un = U / row_norms
    # k-means聚类
    km = KMeans(n_clusters=2, n_init=10).fit(Un)
    print("基于谱嵌入+k-means的标签:", km.labels_)

# 当脚本直接运行时，执行演示函数
if __name__ == "__main__":
    demo_small_example()