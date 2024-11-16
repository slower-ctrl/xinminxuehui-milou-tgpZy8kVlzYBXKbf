
# 模型剪枝


模型剪枝：将模型中不重要的权重和分支裁剪掉。将权重矩阵中一部分元素变为零元素。


![image-20241113084517922](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741674-141522078.png)


减去不重要的突触（Synapses）或神经元（Neurons）。


## 剪枝类型


### 非结构化剪枝


非结构化剪枝：破坏了原有模型的结构。


怎么做：
非结构化剪枝并不关心权重在网络中的位置，只是根据某种标准（例如，权重的绝对值大小）来决定是否移除这个权重。移除权重后，剩下的权重分布是稀疏的，即大多数权重为零。


实际情况：
非结构化剪枝能**极大降低**模型的参数量和理论计算量，但是**现有硬件架构的计算方式无法对其进行加速**，通常需要特殊的硬件或软件支持来有效利用结果模型的稀疏性。所以在**实际运行速度上得不到提升**，需要设计特定的硬件才可能加速。


### 结构化剪枝


结构化剪枝则更加关注模型的组织结构，这种剪枝方法可能涉及到移除整个神经元、卷积核、层或者更复杂的结构。


通常以filter或者整个网络层为基本单位进行剪枝。


一个filter被剪枝，那么其前一个特征图和下一个特征图都会发生相应的变化，但是模型的结构却没有被破坏，仍然能够通过 GPU 或其他硬件来加速。


### 半结构化剪枝


这种剪枝方法可能涉及到移除整个神经元或过滤器的一部分，而不是全部。


通常的做法是按某种规则对结构中的一部分进行剪枝，比如在某个维度上做非结构化剪枝，而在其他维度上保持结构化。


## 剪枝范围


局部剪枝：关注的是模型中的单个权重或参数。这种剪枝方法通常针对模型中的每个权重进行评估，然后决定是否将其设置为零。


全局剪枝：全局剪枝则考虑模型的整体结构和性能。这种剪枝方法可能会移除整个神经元、卷积核、层或者更复杂的结构，如卷积核组。全局剪枝通常需要对模型的整体结构有深入的理解，并且可能涉及到模型架构的重设计。这种方法可能会对模型的最终性能产生更大的影响，因为它改变了模型的整体特征提取能力。


# 剪枝粒度


按照剪枝粒度进行划分，剪枝可分为细粒度剪枝（Fine\-grained Pruning）、基于模式的剪枝（Pattern\-based Pruning）、向量级剪枝（Vector\-level Pruning）、内核级剪枝（Kernel\-level Pruning）与通道级剪枝（Channel\-level Pruning）。


如下图所示，展示了从细粒度剪枝到通道级的剪枝，剪枝越来越规则和结构化。


![image-20241113092055218](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741593-1047064045.png)


## 细粒度剪枝



```
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print("{} 函数的执行时间为：{:.8f} 秒".format(func.__name__, execution_time))
        return result
    return wrapper


# 创建一个可视化2维矩阵函数，将值为0的元素与其他区分开（用于显示剪枝效果）
def plot_tensor(tensor, title):
    # 创建一个新的图像和轴
    fig, ax = plt.subplots()

    # 使用 CPU 上的数据，转换为 numpy 数组，并检查相等条件，设置颜色映射
    ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # 遍历矩阵中的每个元素并添加文本标签
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            text = ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    # 显示图像
    plt.show()


def test_plot_tensor():
    weight = torch.tensor([[-0.46, -0.40, 0.39, 0.19, 0.37],
                           [0.00, 0.40, 0.17, -0.15, 0.16],
                           [-0.20, -0.23, 0.36, 0.25, 0.03],
                           [0.24, 0.41, 0.07, 0.00, -0.15],
                           [0.48, -0.09, -0.36, 0.12, 0.45]])
    plot_tensor(weight, 'weight')


# 细粒度剪枝方法1
@timing_decorator
def _fine_grained_prune(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    遍历矩阵中每个元素，如果元素值小于阈值，则将其设置为0。
    参数太大的话，遍历会影响到速度，下面将介绍在剪枝中常用的一种方法，即使用mask掩码矩阵来实现。
    :param tensor: 输入张量，包含需要剪枝的权重。
    :param threshold: 阈值，用于判断权重的大小。
    :return: 剪枝后的张量。
    """
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            if tensor[i, j] < threshold:
                tensor[i][j] = 0
    return tensor


# 细粒度剪枝方法2
@timing_decorator
def fine_grained_prune(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    创建一个掩码张量，指示哪些权重不应被剪枝（应保持非零）。
    :param tensor: 输入张量，待剪枝的权重。
    :param threshold: 阈值，用于判断权重的大小。
    :return: 剪枝后的张量。
    """
    mask = torch.gt(tensor, threshold)
    tensor.mul_(mask)
    return tensor


if __name__ == '__main__':
    # 创建一个矩阵weight
    weight = torch.rand(8, 8)
    plot_tensor(weight, '剪枝前weight')
    pruned_weight1 = _fine_grained_prune(weight, 0.5)
    plot_tensor(weight, '细粒度剪枝后weight1')
    pruned_weight2 = fine_grained_prune(weight, 0.5)
    plot_tensor(pruned_weight2, '细粒度剪枝后weight2')


```

在掩码剪枝中，一旦生成了掩码矩阵（通常是一个与权重矩阵同形状的二进制矩阵），你可以直接使用掩码与权重进行元素级别的运算，而无需再遍历整个矩阵。


这使得剪枝的过程可以通过向量化操作来加速，尤其是在使用 GPU 时，向量化和矩阵操作比逐元素遍历更高效。


## 基于模式的剪枝



```
import torch
import matplotlib.pyplot as plt
from itertools import permutations

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 创建一个可视化2维矩阵函数，将值为0的元素与其他区分开（用于显示剪枝效果）
def plot_tensor(tensor, title):
    # 创建一个新的图像和轴
    fig, ax = plt.subplots()

    # 使用 CPU 上的数据，转换为 numpy 数组，并检查相等条件，设置颜色映射
    ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # 遍历矩阵中的每个元素并添加文本标签
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            text = ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    # 显示图像
    plt.show()


def reshape_1d(tensor, m):
    # 转换成列为m的格式，若不能整除m则填充0
    if tensor.shape[1] % m > 0:
        mat = torch.FloatTensor(tensor.shape[0], tensor.shape[1] + (m - tensor.shape[1] % m)).fill_(0)
        mat[:, : tensor.shape[1]] = tensor
        return mat.view(-1, m)
    else:
        return tensor.view(-1, m)


def compute_valid_1d_patterns(m, n):
    patterns = torch.zeros(m)
    patterns[:n] = 1
    valid_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    return valid_patterns


def compute_mask(tensor, m, n):
    # tensor={tensor(8,8)}
    # 计算所有可能的模式  patterns={tensor(6,4)}
    patterns = compute_valid_1d_patterns(m, n)
    # 找到m:n最好的模式
    # mask={tensor(16,4)}
    mask = torch.IntTensor(tensor.shape).fill_(1).view(-1, m)  # 使用 -1 让 PyTorch 自动推导某一维的大小
    # mat={tensor(16,4)}
    mat = reshape_1d(tensor, m)
    # pmax={tensor(16,)} 16x4 4x6 = 16x6 -> argmax = 16
    pmax = torch.argmax(torch.matmul(mat.abs(), patterns.t()), dim=1)
    mask[:] = patterns[pmax[:]]  # 选取最好的模式
    mask = mask.view(tensor.shape)  # 得到8x8掩码矩阵
    return mask


def pattern_pruning(tensor, m, n):
    mask = compute_mask(weight, m, n)
    tensor.mul_(mask)
    return tensor


if __name__ == '__main__':
    # 创建一个矩阵weight
    weight = torch.rand(8, 8)
    plot_tensor(weight, '剪枝前weight')
    pruned_weight = pattern_pruning(weight, 4, 2)
    plot_tensor(pruned_weight, '剪枝后weight')


```

**基于模式的剪枝（Pattern\-based Pruning）** 是一种通过预定义的模式来决定剪枝的权重的剪枝方法。在这种方法中，剪枝不再是基于单个权重的大小或者梯度，而是基于一组预定义的剪枝模式，模式决定了哪些权重需要被剪枝，哪些需要保留。


### 1\. **概念解释**


以 **NVIDIA 4:2 剪枝** 为例，假设我们有一个由 4 个权重组成的单元（例如，4 个过滤器、4 个神经元等），我们选择其中 2 个权重进行剪枝，也就是说，将 2 个权重置为 0，而保留剩余的 2 个权重。


* **模式（Pattern）**：我们可以定义 6 种可能的剪枝模式，表示从 4 个权重中选择 2 个权重为 0 的方式。例如，如果我们用 `1` 表示保留的权重，用 `0` 表示被剪枝的权重，那么 6 种可能的模式如下：
	+ `1100`
	+ `1010`
	+ `1001`
	+ `0110`
	+ `0101`
	+ `0011`


每一种模式都表示剪枝过程中保留的权重和被剪枝的权重的组合。


### 2\. **权重矩阵转换与模式匹配**


为了应用这些剪枝模式，我们首先需要将权重矩阵变换为一个适合进行模式匹配的格式：


1. **将权重矩阵变换为 `nx4` 形状**：假设原始的权重矩阵是一个 `n x 4` 的矩阵，其中 `n` 表示样本数量或特征维度，而 `4` 表示每个样本的 4 个权重。
2. **应用模式**：为了与预定义的 6 种模式进行匹配，我们需要计算每个样本在这 4 个权重中符合哪一种模式。计算的结果是一个 `n x 6` 的矩阵，表示每个样本与每种模式的匹配程度（例如，可以是权重的总和、或者其他一些指标，如均值、方差等）。
3. **选择最佳模式**：对于每个样本，我们通过 `argmax` 操作，在 `n` 维度上选择最大值的索引，表示该样本与某一种模式最匹配。得到的索引对应于 6 种模式之一。
4. **构建掩码（Mask）矩阵**：最后，根据选择的模式索引，我们将这些索引映射到对应的模式上，构建一个掩码矩阵。该掩码矩阵会告诉我们哪些权重应该被保留，哪些应该被剪枝。


### 3\. **详细步骤解释**


让我们通过一个具体的例子来详细理解这个过程：


假设我们有一个 `n x 4` 的权重矩阵 `W`，每行是一个 4 维的权重向量：



```
W = [
    [0.5, 0.2, 0.3, 0.8],  # 第一个样本的4个权重
    [0.4, 0.1, 0.7, 0.6],  # 第二个样本的4个权重
    [0.6, 0.5, 0.4, 0.3]   # 第三个样本的4个权重
]

```

然后，我们定义了 6 种剪枝模式，如下：



```
Pattern 1: 1100 (保留第 1 和第 2 个权重)
Pattern 2: 1010 (保留第 1 和第 3 个权重)
Pattern 3: 1001 (保留第 1 和第 4 个权重)
Pattern 4: 0110 (保留第 2 和第 3 个权重)
Pattern 5: 0101 (保留第 2 和第 4 个权重)
Pattern 6: 0011 (保留第 3 和第 4 个权重)

```

1. **计算与模式匹配**：我们可以通过计算每个样本在 4 个权重中的值与每种模式的相似性来得出一个 `n x 6` 的矩阵。例如，计算每个样本的权重和每种模式的匹配度，可能采用简单的加和或者其他复杂的指标。


假设我们对每种模式计算权重的总和，结果如下：



```
match_matrix = [
    [1.0, 0.8, 0.7, 1.0, 0.9, 0.6],  # 第一个样本与每个模式的匹配度
    [0.9, 0.7, 1.1, 0.9, 1.2, 0.5],  # 第二个样本与每个模式的匹配度
    [1.1, 1.0, 0.9, 1.0, 1.0, 1.1]   # 第三个样本与每个模式的匹配度
]

```
2. **选择最佳模式**：通过对 `match_matrix` 进行 `argmax` 操作，我们可以选择每个样本与哪一种模式最匹配：



```
best_pattern_indices = [0, 4, 5]  # 对应样本 1 最匹配模式 1，样本 2 最匹配模式 5，样本 3 最匹配模式 6

```
3. **填充掩码（Mask）矩阵**：根据每个样本选择的模式，我们填充掩码矩阵。例如，样本 1 选择了模式 1（即 `1100`），样本 2 选择了模式 5（即 `0101`），样本 3 选择了模式 6（即 `0011`）。


最终得到的掩码矩阵 `mask` 就是：



```
mask = [
    [1, 1, 0, 0],  # 样本 1 对应模式 1
    [0, 1, 0, 1],  # 样本 2 对应模式 5
    [0, 0, 1, 1]   # 样本 3 对应模式 6
]

```
4. **应用掩码到权重矩阵**：将这个掩码矩阵与权重矩阵进行逐元素相乘，就完成了剪枝操作。


### 4\. **总结**


基于模式的剪枝通过以下步骤提升了效率：


1. **预定义模式**：定义剪枝模式，而不是针对每个权重进行逐一选择。
2. **模式匹配**：通过计算每个样本与模式的匹配度，并选择最佳匹配的模式。
3. **掩码应用**：通过掩码矩阵直接将剪枝信息应用到权重矩阵中，避免了频繁的元素遍历和修改操作。


相比于逐个权重剪枝，基于模式的剪枝能够更高效地处理剪枝任务，特别是在大规模的模型中。


## 向量级别剪枝



```
import torch
import matplotlib.pyplot as plt
from itertools import permutations

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 创建一个可视化2维矩阵函数，将值为0的元素与其他区分开（用于显示剪枝效果）
def plot_tensor(tensor, title):
    # 创建一个新的图像和轴
    fig, ax = plt.subplots()

    # 使用 CPU 上的数据，转换为 numpy 数组，并检查相等条件，设置颜色映射
    ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # 遍历矩阵中的每个元素并添加文本标签
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            text = ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    # 显示图像
    plt.show()
# 剪枝某个点所在的行与列
def vector_pruning(weight, point):
    row, col = point
    prune_weight = weight.clone()
    prune_weight[row, :] = 0
    prune_weight[:, col] = 0
    return prune_weight
if __name__ == '__main__':
    weight = torch.rand(8, 8)
    point = (1, 1)
    prune_weight = vector_pruning(weight, point)
    plot_tensor(prune_weight, '向量级剪枝后weight')

```

## 卷积核级别剪枝



```
tensor = torch.rand((3, 10, 4, 5))  # 3 batch size, 10 channels, 4 height, 5 width

```

![image-20241113131650800](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741666-1925847991.png)
10个通道则1个过滤器有10个卷积核。


![image-20241113132059624](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741500-360136671.png)
红色的部分代表从中去掉一个卷积核。



```
import torch
import matplotlib.pyplot as plt
from itertools import permutations

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 定义可视化4维张量的函数
def visualize_tensor(tensor, title, batch_spacing=3):
    fig = plt.figure()  # 创建一个新的matplotlib图形
    ax = fig.add_subplot(111, projection='3d')  # 向图形中添加一个3D子图

    # 遍历张量的批次维度
    for batch in range(tensor.shape[0]):
        # 遍历张量的通道维度
        for channel in range(tensor.shape[1]):
            # 遍历张量的高度维度
            for i in range(tensor.shape[2]):
                # 遍历张量的宽度维度
                for j in range(tensor.shape[3]):
                    # 计算条形的x位置，考虑到不同批次间的间隔
                    x = j + (batch * (tensor.shape[3] + batch_spacing))
                    y = i  # 条形的y位置，即张量的高度维度
                    z = channel  # 条形的z位置，即张量的通道维度
                    # 如果张量在当前位置的值为0，则设置条形颜色为红色，否则为绿色
                    color = 'red' if tensor[batch, channel, i, j] == 0 else 'green'
                    # 绘制单个3D条形
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=color, edgecolor='black', alpha=0.9)

    ax.set_title(title)  # 设置3D图形的标题
    ax.set_xlabel('Width')  # 设置x轴标签，对应张量的宽度维度
    ax.set_ylabel('Height')  # 设置y轴标签，对应张量的高度维度
    ax.set_zlabel('Channel')  # 设置z轴标签，对于张量的通道维度
    ax.set_zlim(ax.get_zlim()[::-1])  # 反转z轴方向
    ax.zaxis.labelpad = 15  # 调整z轴标签的填充

    plt.show()  # 显示图形


def prune_conv_layer(conv_layer, title, percentile=0.2, ):
    prune_layer = conv_layer.clone()

    # 计算每个kernel的L2范数
    l2_norm = torch.norm(prune_layer, p=2, dim=(-2, -1), keepdim=True)
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    prune_layer = prune_layer * mask.float()

    visualize_tensor(prune_layer, title=title)


if __name__ == '__main__':
    # 使用PyTorch创建一个张量
    tensor = torch.rand((3, 10, 4, 5))  # 3 batch size, 10 channels, 4 height, 5 width
    # 调用函数进行剪枝
    pruned_tensor = prune_conv_layer(tensor, 'Kernel级别剪枝')


```

## 过滤器级别剪枝


![image-20241113132441778](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741668-1277352921.png)
相当于这一组卷积核的结果都不要了。



```
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 定义可视化4维张量的函数
def visualize_tensor(tensor, title, batch_spacing=3):
    fig = plt.figure()  # 创建一个新的matplotlib图形
    ax = fig.add_subplot(111, projection='3d')  # 向图形中添加一个3D子图

    # 遍历张量的批次维度
    for batch in range(tensor.shape[0]):
        # 遍历张量的通道维度
        for channel in range(tensor.shape[1]):
            # 遍历张量的高度维度
            for i in range(tensor.shape[2]):
                # 遍历张量的宽度维度
                for j in range(tensor.shape[3]):
                    # 计算条形的x位置，考虑到不同批次间的间隔
                    x = j + (batch * (tensor.shape[3] + batch_spacing))
                    y = i  # 条形的y位置，即张量的高度维度
                    z = channel  # 条形的z位置，即张量的通道维度
                    # 如果张量在当前位置的值为0，则设置条形颜色为红色，否则为绿色
                    color = 'red' if tensor[batch, channel, i, j] == 0 else 'green'
                    # 绘制单个3D条形
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=color, edgecolor='black', alpha=0.9)

    ax.set_title(title)  # 设置3D图形的标题
    ax.set_xlabel('Width')  # 设置x轴标签，对应张量的宽度维度
    ax.set_ylabel('Height')  # 设置y轴标签，对应张量的高度维度
    ax.set_zlabel('Channel')  # 设置z轴标签，对于张量的通道维度
    ax.set_zlim(ax.get_zlim()[::-1])  # 反转z轴方向
    ax.zaxis.labelpad = 15  # 调整z轴标签的填充

    plt.show()  # 显示图形


def prune_conv_layer(conv_layer, prune_method, title="", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()

    l2_norm = None
    mask = None

    # 计算每个Filter的L2范数
    l2_norm = torch.norm(prune_layer, p=2, dim=(1, 2, 3), keepdim=True)
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    prune_layer = prune_layer * mask.float()

    visualize_tensor(prune_layer, title=prune_method)

if __name__ == '__main__':
    # 使用PyTorch创建一个张量
    tensor = torch.rand((3, 10, 4, 5))

    # 调用函数进行剪枝

    pruned_tensor = prune_conv_layer(tensor, 'Filter级别剪枝', vis=True)

```

## 通道级别剪枝


![image-20241113132703072](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741664-1596756929.png)

```
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 定义可视化4维张量的函数
def visualize_tensor(tensor, title, batch_spacing=3):
    fig = plt.figure()  # 创建一个新的matplotlib图形
    ax = fig.add_subplot(111, projection='3d')  # 向图形中添加一个3D子图

    # 遍历张量的批次维度
    for batch in range(tensor.shape[0]):
        # 遍历张量的通道维度
        for channel in range(tensor.shape[1]):
            # 遍历张量的高度维度
            for i in range(tensor.shape[2]):
                # 遍历张量的宽度维度
                for j in range(tensor.shape[3]):
                    # 计算条形的x位置，考虑到不同批次间的间隔
                    x = j + (batch * (tensor.shape[3] + batch_spacing))
                    y = i  # 条形的y位置，即张量的高度维度
                    z = channel  # 条形的z位置，即张量的通道维度
                    # 如果张量在当前位置的值为0，则设置条形颜色为红色，否则为绿色
                    color = 'red' if tensor[batch, channel, i, j] == 0 else 'green'
                    # 绘制单个3D条形
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=color, edgecolor='black', alpha=0.9)

    ax.set_title(title)  # 设置3D图形的标题
    ax.set_xlabel('Width')  # 设置x轴标签，对应张量的宽度维度
    ax.set_ylabel('Height')  # 设置y轴标签，对应张量的高度维度
    ax.set_zlabel('Channel')  # 设置z轴标签，对于张量的通道维度
    ax.set_zlim(ax.get_zlim()[::-1])  # 反转z轴方向
    ax.zaxis.labelpad = 15  # 调整z轴标签的填充

    plt.show()  # 显示图形


def prune_conv_layer(conv_layer, prune_method, title="", percentile=0.2, vis=True):
    prune_layer = conv_layer.clone()

    l2_norm = None
    mask = None

    # 计算每个channel的L2范数
    l2_norm = torch.norm(prune_layer, p=2, dim=(0, 2, 3), keepdim=True)
    threshold = torch.quantile(l2_norm, percentile)
    mask = l2_norm > threshold
    prune_layer = prune_layer * mask.float()

    visualize_tensor(prune_layer, title=prune_method)


# 使用PyTorch创建一个张量
tensor = torch.rand((3, 10, 4, 5))

# 调用函数进行剪枝

pruned_tensor = prune_conv_layer(tensor, 'Channel级别剪枝', vis=True)

```

所有级别剪枝对比：



```
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码


# 创建一个可视化2维矩阵函数，将值为0的元素与其他区分开（用于显示剪枝效果）
def plot_tensor(tensor, title):
    # 创建一个新的图像和轴
    fig, ax = plt.subplots()

    # 使用 CPU 上的数据，转换为 numpy 数组，并检查相等条件，设置颜色映射
    ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # 遍历矩阵中的每个元素并添加文本标签
    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[0]):
            text = ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    # 显示图像
    plt.show()


# 剪枝某个点所在的行与列
def vector_pruning(weight, point):
    row, col = point
    prune_weight = weight.clone()
    prune_weight[row, :] = 0
    prune_weight[:, col] = 0
    return prune_weight


if __name__ == '__main__':
    weight = torch.rand(8, 8)
    point = (1, 1)
    prune_weight = vector_pruning(weight, point)
    plot_tensor(prune_weight, '向量级剪枝后weight')


```

# 剪枝标准


模型剪枝之所以有效，主要是因为它能够识别并移除那些对模型性能影响较小的参数，从而减少模型的复杂性和计算成本。


其背后的理论依据主要集中在以下几个方面：


* 彩票假说：该假说认为，在随机初始化的大型神经网络中，存在一个子网络，如果独立训练，可以达到与完整网络相似的性能。这表明网络中并非所有部分都对最终性能至关重要，从而为剪枝提供了理论支持。
* 网络稀疏性：研究发现，许多深度神经网络参数呈现出稀疏性，即大部分参数值接近于零。这种稀疏性启发了剪枝技术，即通过移除这些非显著的参数来简化模型。
* 剪枝的一个重要理论来源是正则化，特别是L1正则化，它鼓励网络学习稀疏的参数分布。稀疏化的模型更容易进行剪枝，因为许多权重接近于零，可以安全移除。
* 权重的重要性：剪枝算法通常基于权重的重要性来决定是否剪枝。**权重的重要性**可以通过多种方式评估，例如**权重的大小**、**权重对损失函数的梯度**、或者**权重对输入的激活情况**等。


怎么确定要减掉哪些呢？这就涉及到剪枝标准。


## 基于权重大小


这种剪枝方法基于一个假设，即**权重的绝对值越小，该权重对模型的输出影响越小**，因此移除它们对模型性能的影响也会较小。


![image-20241113133840952](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741178-1931757436.png)


这里也就是计算每个格子中权重的绝对值，绝对值大的保留，小的移除。


L1和L2正则化是机器学习中常用的正则化技术，它们通过在损失函数中添加额外的惩罚项来防止模型过拟合。


### L1和L2正则化


[深入理解L1、L2正则化 \- ZingpLiu \- 博客园](https://github.com)


**正则化**是机器学习中对原始损失函数引入额外信息，以便防止过拟合和提高模型泛化性能的一类方法的统称。也就是目标函数变成了**原始损失函数\+额外项**，常用的额外项一般有两种，英文称作ℓ1−normℓ1−norm和ℓ2−normℓ2−norm，中文称作**L1正则化**和**L2正则化**，或者L1范数和L2范数（实际是L2范数的平方）。


正则化技术（如L1和L2）通过**限制模型的权重**来控制模型的复杂度，避免模型过拟合。对于一个包含多个特征的模型，如果所有特征的权重都很大，说明模型可能对每个特征都高度依赖，这样容易在训练集上过拟合。


我们将L1或L2正则化加入到损失函数中，目的是惩罚那些过大的权重。**惩罚项**的作用是增加模型训练时的成本，从而迫使模型尽可能避免使用过大的权重值。


* **惩罚**表示当模型的权重过大时，正则化项会增加损失函数的值，使得模型更倾向于选择较小的权重。这就像给模型设定了一种惩罚规则，避免它在训练过程中“过度自信”地依赖某些特征。
* **控制复杂度**：惩罚项的加入，限制了模型参数的大小，减少了模型对训练数据的过拟合。


在没有正则化的情况下，模型仅仅关注最小化预测误差（即损失函数），它可能会通过对某些特征赋予很大的权重来达到最小化损失，这会导致过拟合。加入正则化项后，损失函数不仅考虑预测误差，还会考虑模型的复杂度，这样就能够找到一个平衡点，避免模型过度拟合。


**L1 正则化**


![image-20241115103931172](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741168-1179127476.png)


L1正则化的加入项是绝对值之和，这意味着它可以**产生稀疏解**——有些权重会被压缩为零，导致对应的特征完全被剔除。这样做的好处是，模型变得更加简洁和可解释，同时可以进行**特征选择**，仅保留那些最重要的特征。


**L2 正则化**


![image-20241115103945543](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162740889-25642560.png)
L2正则化倾向于使得权重变小，但不会将权重压缩为零。它的作用是让模型更稳定，减少对某些特征的过度依赖，但不会像L1正则化那样进行特征选择。


## L1、L2正则化剪枝


L1和L2正则化基本思想是以行为单位，计算每行的重要性，移除权重中那些重要性较小的行。


L1行剪枝：


![image-20241115104213213](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741505-1763954383.png)


L2行剪枝：


![image-20241115104246680](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741272-40069210.png)


### LeNet



```
# 定义一个LeNet网络
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

```

* **卷积层 (`conv1`)**：


	+ `nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)`
	+ 输入的图像通道数为 1（灰度图像），输出 6 个特征图，每个特征图大小为 28x28（5x5 卷积核，图像尺寸会变小）。
* **卷积层 (`conv2`)**：


	+ `nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)`
	+ 输入 6 个特征图，输出 16 个特征图。每个特征图大小为 10x10（再次进行 5x5 卷积）。
* **池化层 (`maxpool`)**：


	+ `nn.MaxPool2d(kernel_size=2, stride=2)`
	+ 2x2 的最大池化操作，步长为 2，这会将每个特征图的尺寸缩小一半。
* **全连接层 (`fc1`, `fc2`, `fc3`)**：


	+ `nn.Linear(in_features=16 * 4 * 4, out_features=120)`
	+ 第一个全连接层，将 16 个 4x4 的特征图展平为 1D 向量，输入 256 个特征，输出 120 个神经元。
	+ `nn.Linear(in_features=120, out_features=84)`
	+ 第二个全连接层，输入 120 个神经元，输出 84 个神经元。
	+ `nn.Linear(in_features=84, out_features=num_classes)`
	+ 第三个全连接层，输出最终的分类结果，这里 `num_classes=10` 对应 MNIST 数据集的 10 个数字类别。


**`forward` 方法**：


* 该方法定义了模型的前向传播过程。
* **第一层卷积和池化**：


	+ `x = self.maxpool(F.relu(self.conv1(x)))`
	+ 对输入 `x` 进行卷积（`conv1`），然后通过 ReLU 激活函数，再通过最大池化层（`maxpool`）。
* **第二层卷积和池化**：


	+ `x = self.maxpool(F.relu(self.conv2(x)))`
	+ 同样，对卷积（`conv2`）的输出进行 ReLU 激活和池化。
* **展平**：


	+ `x = x.view(x.size()[0], -1)`
	+ 将经过卷积和池化后的输出展平为 1D 向量，为进入全连接层做准备。`x.size()[0]` 表示批次大小，`-1` 表示自动计算其余维度。
* **全连接层**：


	+ `x = F.relu(self.fc1(x))`
	+ `x = F.relu(self.fc2(x))`
	+ `x = self.fc3(x)`
	+ 使用 ReLU 激活函数处理全连接层的输出，并最终得到分类结果。


### 基于L1权重大小的剪枝



```
@torch.no_grad()
def prune_l1(weight, percentile=0.5):
    # 计算权重个数 2400=16*6*5*5
    num_elements = weight.numel()

    # 计算值为0的数量 num_zeros=200
    num_zeros = round(num_elements * percentile)
    # 计算weight的重要性 tensor{(16,6,5,5)}
    importance = weight.abs()
    # 计算裁剪阈值 tensor(0.0451, device='cuda:0')
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # 计算mask (小于阈值的设置为False，大于阈值的设置为True)
    mask = torch.gt(importance, threshold)

    # 计算mask后的weight
    weight.mul_(mask)
    return weight

```

这段代码是一个 **L1 正则化剪枝（pruning）** 函数，目的是通过 **裁剪** （prune）掉网络中一些不重要的权重，以减小模型的复杂度，通常用于模型压缩和加速推理过程。


* **`@torch.no_grad()`**：
这个装饰器告诉 PyTorch 在该函数执行时不计算梯度。即使在该函数内部做了修改（如 `weight.mul_(mask)`），也不会追踪这些操作的梯度。这通常用于推理或一些不需要梯度计算的操作，避免额外的内存开销。


**参数**：


* **`weight`**：
这是模型某层的权重张量（tensor），通常是一个二维张量，对应于卷积层或全连接层的权重矩阵。
* **`percentile`**：
这是一个介于 0 到 1 之间的浮动值，表示要裁剪掉的权重的比例。例如，`percentile=0.5` 表示剪掉最小的一半权重。


**详细步骤**：


1. **计算权重的元素数量**：
`num_elements = weight.numel()`
这行代码计算 `weight` 张量中元素的总数量（即权重的个数）。
2. **计算需要剪去的权重数量**：
`num_zeros = round(num_elements * percentile)`
这里计算需要剪去的权重数量。`percentile` 决定了要剪去的权重占比，`num_zeros` 是该占比对应的权重数量。
3. **计算权重的“重要性”**：
`importance = weight.abs()`
这一步通过对权重取 **绝对值** 来衡量其“重要性”。一般来说，L1 范数（绝对值）越小的权重，对模型的影响越小，因此可以认为它们较不重要。
4. **计算裁剪的阈值**：
`threshold = importance.view(-1).kthvalue(num_zeros).values`
将 `importance` 展平为一维向量（`view(-1)`），然后通过 `kthvalue` 函数找到第 `num_zeros` 小的值。这个值即为裁剪阈值，表示剪去比这个值小的权重。
5. **计算掩码（Mask）**：
`mask = torch.gt(importance, threshold)`
这行代码生成一个布尔值的掩码（mask），其中 `True` 表示该权重的重要性大于阈值，`False` 表示该权重的重要性小于阈值。`torch.gt` 是“大于”的意思。
6. **应用掩码进行剪枝**：
`weight.mul_(mask)`
使用 `mask` 来筛选权重，`True` 的位置保持原值，`False` 的位置会被设为零。`mul_` 是对 `weight` 进行原地（in\-place）乘法操作，即在原始权重张量上直接进行修改。
7. **返回剪枝后的权重**：
`return weight`
最终返回经过剪枝后的权重。


**总结**：


这个函数的核心思路是：


1. 计算每个权重的“重要性”，通过其绝对值（L1 范数）衡量。
2. 根据设置的 `percentile` 参数，裁剪掉最不重要的权重。
3. 使用一个布尔掩码（mask）将不重要的权重置为零，从而实现模型的稀疏化。


剪枝后分布：


![image-20241115152630415](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741096-40947431.png)
![image-20241115152614186](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162740904-83275396.png)
* x 轴代表 **权重值的大小**，表示模型中每个权重参数的数值范围。
* y 轴表示 **权重值的密度**（density），即单位区间内权重的数量。


减少了一半权重参数：


![image-20241115153138672](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741357-1612057942.png)
### 基于L2权重大小的剪枝



```
@torch.no_grad()
def prune_l2(weight, percentile=0.5):
    num_elements = weight.numel()

    # 计算值为0的数量
    num_zeros = round(num_elements * percentile)
    # 计算weight的重要性（使用L2范数，即各元素的平方）
    importance = weight.pow(2)  # 这里和上面不同
    # 计算裁剪阈值
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # 计算mask
    mask = torch.gt(importance, threshold)
    
    # 计算mask后的weight
    weight.mul_(mask)
    return weight

# 裁剪fc1层（全连接）
weight_pruned = prune_l2(model.fc1.weight, percentile=0.4)  # 裁剪40%
# 替换原有model层
model.fc1.weight.data = weight_pruned
# 列出weight直方图
plot_weight_distribution(model)

```

裁剪后分布 ：


![image-20241115154244048](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162740961-34845613.png)
![image-20241115154222146](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162740878-1698685803.png)
减少了40%参数：


![image-20241115154741661](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741344-732118060.png)
## 基于梯度大小


核心思想：**在模型训练过程中，权重的梯度反映了权重对输出损失的影响程度，较大的梯度表示权重对输出损失的影响较大，因此较重要；较小的梯度表示权重对输出损失的影响较小，因此较不重要。**通过去除较小梯度的权重，可以减少模型的规模，同时保持模型的准确性。


对比以权值大小为重要性依据的剪枝算法：以人脸识别为例，在人脸的诸多特征中，眼睛的细微变化如颜色、大小、形状，对于人脸识别的结果有很大影响。对应到深度网络中的权值，**即使权值本身很小，但是它的细微变化对结果也将产生很大的影响，这类权值是不应该被剪掉的。**梯度是计算损失函数对权值的偏导数，反映了损失对权值的敏感程度。基于梯度大小的剪枝算法是一种通过分析模型中权重梯度的方法，来判断权重的重要性，并去除较小梯度的权重的剪裁方法。



```
import copy
import math
import random
import time

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# 设置 matplotlib 使用支持负号的字体
plt.rcParams['font.family'] = 'DejaVu Sans'


# 绘制权重分布图
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # 删除多余的子图
    fig.delaxes(axes[1][2])

    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color='green', alpha=0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color='green', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


# 为避免前面的操作影响后续结果，重新定义一个LeNet网络，和前面一致
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)

# 加载梯度信息
gradients = torch.load('./model_gradients.pt')
# 加载参数信息
checkpoint = torch.load('./model.pt')
# 加载状态字典到模型
model.load_state_dict(checkpoint)


# 修剪整个模型的权重，传入整个模型
def gradient_magnitude_pruning(model, percentile):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 当梯度的绝对值大于或等于这个阈值时，权重会被保留。
            mask = torch.abs(gradients[name]) >= percentile
            param.data *= mask.float()


# 修剪局部模型权重，传入某一层的权重
@torch.no_grad()
def gradient_magnitude_pruning(weight, gradient, percentile=0.5):
    num_elements = weight.numel()
    # 计算值为0的数量
    num_zeros = round(num_elements * percentile)
    # 计算weight的重要性（使用L1范数）
    importance = gradient.abs()
    # 计算裁剪阈值
    threshold = importance.view(-1).kthvalue(num_zeros).values
    # 计算mask
    mask = torch.gt(importance, threshold)
    # 确保mask和weight在同一设备上
    mask = mask.to(weight.device)
    # 计算mask后的weight
    weight.mul_(mask)
    return weight


if __name__ == '__main__':
    # 使用示例，这里以fc2层的权重为例
    percentile = 0.5
    gradient_magnitude_pruning(model.fc2.weight, gradients['fc2.weight'], percentile)
    # 列出weight直方图
    plot_weight_distribution(model)

```

![image-20241115160423227](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741199-1358439407.png)
![image-20241115161534838](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741021-644521569.png)
## 基于尺度


[通俗理解 Batch Normalization（含代码） \- 知乎](https://github.com)


Network Slimming提出了一种基于尺度(Scaling\-based)的剪枝方法。这种方法：**剪枝整个通道**
识别并剪枝那些对模型输出影响不大的整个通道（即一组特征映射），而不是单个权重。


在标准的CNN训练中，批归一化（BN）层通常用于加速训练并提高模型的泛化能力。该方法利用BN层中的缩放因子（γ）来实现稀疏性。这些缩放因子原本用于调节BN层输出的尺度，但在该方法中，它们被用来指示每个通道的重要性。在训练过程中，通过在损失函数中添加一个L1正则化项来鼓励通道的缩放因子趋向于零。这样，不重要的通道的缩放因子将变得非常小，从而可以被识别并剪枝。


## 基于二阶


基于二阶（Second\-Order\-based）的剪枝方法中最具代表性的是最优脑损伤（Optimal Brain Damage，OBD）。**OBD通过最小化由于剪枝突触引入的损失函数误差，利用二阶导数信息来评估网络中每个权重的重要性，然后根据这些评估结果来决定哪些权重可以被剪枝。**


​ 首先，计算网络损失函数相对于权重的Hessian矩阵。Hessian矩阵是一个方阵，其元素是**损失函数相对于网络参数的二阶偏导数**。它提供了关于参数空间中曲线曲率的信息，可以用来判断权重的敏感度。其次，通过分析Hessian矩阵的特征值，可以确定网络参数的重要性。通常，与较大特征值相对应的权重被认为是更重要的，因为它们对损失函数的曲率贡献更大。


![image-20241115110319249](https://img2024.cnblogs.com/blog/2614258/202411/2614258-20241115162741595-861795927.png)
从最后的公式可以看出，OBD方法最后只需要考虑矩阵对角线元素，详细的公式推导过程参考[OBD公式推导](https://github.com)。


# 剪枝频率


### 迭代剪枝


迭代剪枝是一种渐进式的模型剪枝方法，它涉及多个循环的剪枝和微调步骤。这个过程逐步削减模型中的权重，而不是一次性剪除大量的权重。迭代剪枝的基本思想是，通过**逐步移除权重，可以更细致地评估每一次剪枝对模型性能的影响，并允许模型有机会调整其余权重来补偿被剪除的权重**。


迭代剪枝通常遵循以下步骤：


* 训练模型：首先训练一个完整的、未剪枝的模型，使其在训练数据上达到一个良好的性能水平。
* 剪枝：使用一个预定的剪枝策略（例如基于权重大小）来轻微剪枝网络，移除一小部分权重。
* 微调：对剪枝后的模型进行微调，这通常涉及使用原始训练数据集重新训练模型，以恢复由于剪枝引起的性能损失。
* 评估：在验证集上评估剪枝后模型的性能，确保模型仍然能够维持良好的性能。
* 重复：重复步骤2到步骤4，每次迭代剪掉更多的权重，并进行微调，直到达到一个预定的性能标准或剪枝比例。


### 单次剪枝


* 定义：在训练完成后对模型进行一次性的剪枝操作。
* 优点：这种剪枝方法的特点是高效且直接，它不需要在剪枝和再训练之间进行多次迭代。
* 步骤：在One\-shot剪枝中，模型首先被训练到收敛，然后根据某种剪枝标准（如权重的绝对值大小）来确定哪些参数可以被移除。这些参数通常是那些对模型输出影响较小的参数。
* 对比迭代式剪枝：单次剪枝会极大地受到噪声的影响，而迭代式剪枝方法则会好很多，因为它在每次迭代之后只会删除掉少量的权重，然后周而复始地进行其他轮的评估和删除，这就能够在一定程度上减少噪声对于整个剪枝过程的影响。但对于大模型来说，由于微调的成本太高，所以更倾向于使用单次剪枝方法。


# 剪枝时机


### 训练后剪枝


训练后剪枝基本思想是先训练一个模型 ，然后对模型进行剪枝，最后对剪枝后模型进行微调。其核心思想是对模型进行一次训练，以了解哪些神经连接实际上很重要，修剪那些不重要（权重较低）的神经连接，然后再次训练以了解权重的最终值。以下是详细步骤：


* 初始训练：首先，使用标准的反向传播算法训练神经网络。在这个过程中，网络学习到权重（即连接的强度）和网络结构。
* 识别重要连接：在训练完成后，网络已经学习到了哪些连接对模型的输出有显著影响。通常，权重较大的连接被认为是重要的。
* 设置阈值：选择一个阈值，这个阈值用于确定哪些连接是重要的。所有权重低于这个阈值的连接将被视为不重要。
* 剪枝：移除所有权重低于阈值的连接。这通常涉及到将全连接层转换为稀疏层，因为大部分连接都被移除了。
* 重新训练：在剪枝后，网络的容量减小了，为了补偿这种变化，需要重新训练网络。在这个过程中，网络会调整剩余连接的权重，以便在保持准确性的同时适应新的结构。
* 迭代剪枝：剪枝和重新训练的过程可以迭代进行。每次迭代都会移除更多的连接，直到达到一个平衡点，即在不显著损失准确性的情况下尽可能减少连接。


### 训练时剪枝


训练时剪枝基本思想是直接在模型训练过程中进行剪枝，最后对剪枝后模型进行微调。与训练后剪枝相比，连接在训练期间根据其重要性动态停用，但允许权重适应并可能重新激活。训练时剪枝可以产生更有效的模型，因为不必要的连接会尽早修剪，从而可能减少训练期间的内存和计算需求。然而，它需要小心处理，以避免网络结构的突然变化和过度修剪的风险，这可能会损害性能。深度学习中常用到的Dropout其实就是一种训练时剪枝方法，在训练过程中，随机神经元以一定的概率被“dropout”或设置为零。训练时剪枝的训练过程包括以下几个详细步骤，以CNN网络为例：


* 初始化模型参数：首先，使用标准的初始化方法初始化神经网络的权重。
* 训练循环：在每个训练周期（epoch）开始时，使用完整的模型参数对训练数据进行前向传播和反向传播，以更新模型权重。
* 计算重要性：在每个训练周期结束时，计算每个卷积层中所有过滤器的重要性。
* 选择过滤器进行修剪：根据一个预先设定的修剪率，选择重要性最小的过滤器进行修剪。这些过滤器被认为是不重要的，因为它们对模型输出的贡献较小。
* 修剪过滤器：将选择的过滤器的权重设置为零，从而在后续的前向传播中不计算这些过滤器的贡献。
* 重建模型：在修剪过滤器之后，继续进行一个训练周期。在这个阶段，通过反向传播，允许之前被修剪的过滤器的权重更新，从而恢复模型的容量。
* 迭代过程：重复上述步骤，直到达到预定的训练周期数或者模型收敛。


### 训练前剪枝


训练前剪枝基本思想是在模型训练前进行剪枝，然后从头训练剪枝后的模型。这里就要提及到彩票假设，即任何随机初始化的稠密的前馈网络都包含具有如下性质的子网络——在独立进行训练时，初始化后的子网络在至多经过与原始网络相同的迭代次数后，能够达到跟原始网络相近的测试准确率。在彩票假设中，剪枝后的网络不是需要进行微调，而是将“中奖”的子网络重置为网络最初的权重后重新训练，最后得到的结果可以追上甚至超过原始的稠密网络。总结成一句话：随机初始化的密集神经网络包含一个子网络，该子网络经过初始化，以便在单独训练时，在训练最多相同次数的迭代后，它可以与原始网络的测试精度相匹配。


一开始，神经网络是使用预定义的架构和随机初始化的权重创建的。这构成了剪枝的起点。基于某些标准或启发法，确定特定的连接或权重以进行修剪。那么有个问题，我们还没有开始训练模型，那么我们如何知道哪些连接不重要呢？


目前常用的方式一般是在初始化阶段采用随机剪枝的方法。随机选择的连接被修剪，并且该过程重复多次以创建各种稀疏网络架构。这背后的想法是，如果在训练之前以多种方式进行修剪，可能就能够跳过寻找彩票的过程。


### 剪枝时机总结


**训练后剪枝（静态稀疏性）：** 初始训练阶段后的修剪涉及在单独的后处理步骤中从训练模型中删除连接或过滤器。这使得模型能够在训练过程中完全收敛而不会出现任何中断，从而确保学习到的表示得到很好的建立。剪枝后，可以进一步微调模型，以从剪枝过程引起的任何潜在性能下降中恢复过来。训练后的剪枝一般比较稳定，不太可能造成过拟合。适用于针对特定任务微调预训练模型的场景。


**训练时剪枝（动态稀疏）：** 在这种方法中，剪枝作为附加正则化技术集成到优化过程中。在训练迭代期间，根据某些标准或启发方法动态删除或修剪不太重要的连接。这使得模型能够探索不同级别的稀疏性并在整个训练过程中调整其架构。动态稀疏性可以带来更高效的模型，因为不重要的连接会被尽早修剪，从而可能减少内存和计算需求。然而，它需要小心处理，以避免网络结构的突然变化和过度修剪的风险，这可能会损害性能。


**训练前剪枝：** 训练前剪枝涉及在训练过程开始之前从神经网络中剪枝某些连接或权重。优点在于可以更快地进行训练，因为初始模型大小减小了，并且网络可以更快地收敛。然而，它需要仔细选择修剪标准，以避免过于积极地删除重要连接。


# 剪枝比例


假设一个模型有很多层，给定一个全局的剪枝比例，那么应该怎么分配每层的剪枝率呢？主要可以分为两种方法：均匀分层剪枝和非均匀分层剪枝。


* 均匀分层剪枝（Uniform Layer\-Wise Pruning）是指在神经网络的每一层中都应用相同的剪枝率。具体来说，就是对网络的所有层按照统一的标准进行剪枝，无论每一层的权重重要性或梯度如何分布。这种方法实现简单，剪枝率容易控制，但它忽略了每一层对模型整体性能的重要性差异。
* 非均匀分层剪枝（Non\-Uniform Layer\-Wise Pruning）则根据每一层的不同特点来分配不同的剪枝率。例如，可以根据梯度信息、权重的大小、或者其他指标（如信息熵、Hessian矩阵等）来确定每一层的剪枝率。层越重要，保留的参数越多；不重要的层则可以被更大程度地剪枝。如下图3\-9所示，非均匀剪枝往往比均匀剪枝的性能更好。


# 代码


* [剪枝粒度实践](https://github.com)
* [剪枝标准实践](https://github.com)
* [剪枝时机实践](https://github.com)
* [torch中的剪枝算法实践](https://github.com)


  * [模型剪枝](#%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D)
* [剪枝类型](#%E5%89%AA%E6%9E%9D%E7%B1%BB%E5%9E%8B)
* [非结构化剪枝](#%E9%9D%9E%E7%BB%93%E6%9E%84%E5%8C%96%E5%89%AA%E6%9E%9D)
* [结构化剪枝](#%E7%BB%93%E6%9E%84%E5%8C%96%E5%89%AA%E6%9E%9D)
* [半结构化剪枝](#%E5%8D%8A%E7%BB%93%E6%9E%84%E5%8C%96%E5%89%AA%E6%9E%9D)
* [剪枝范围](#%E5%89%AA%E6%9E%9D%E8%8C%83%E5%9B%B4)
* [剪枝粒度](#%E5%89%AA%E6%9E%9D%E7%B2%92%E5%BA%A6)
* [细粒度剪枝](#%E7%BB%86%E7%B2%92%E5%BA%A6%E5%89%AA%E6%9E%9D)
* [基于模式的剪枝](#%E5%9F%BA%E4%BA%8E%E6%A8%A1%E5%BC%8F%E7%9A%84%E5%89%AA%E6%9E%9D)
* [1\. 概念解释](#tid-jJhekf)
* [2\. 权重矩阵转换与模式匹配](#tid-c33SKB)
* [3\. 详细步骤解释](#tid-2MDyDn)
* [4\. 总结](#tid-GMWtWe)
* [向量级别剪枝](#%E5%90%91%E9%87%8F%E7%BA%A7%E5%88%AB%E5%89%AA%E6%9E%9D)
* [卷积核级别剪枝](#%E5%8D%B7%E7%A7%AF%E6%A0%B8%E7%BA%A7%E5%88%AB%E5%89%AA%E6%9E%9D)
* [过滤器级别剪枝](#%E8%BF%87%E6%BB%A4%E5%99%A8%E7%BA%A7%E5%88%AB%E5%89%AA%E6%9E%9D)
* [通道级别剪枝](#%E9%80%9A%E9%81%93%E7%BA%A7%E5%88%AB%E5%89%AA%E6%9E%9D)
* [剪枝标准](#%E5%89%AA%E6%9E%9D%E6%A0%87%E5%87%86):[westworld加速](https://tianchuang88.com)
* [基于权重大小](#%E5%9F%BA%E4%BA%8E%E6%9D%83%E9%87%8D%E5%A4%A7%E5%B0%8F)
* [L1和L2正则化](#l1%E5%92%8Cl2%E6%AD%A3%E5%88%99%E5%8C%96)
* [L1、L2正则化剪枝](#l1l2%E6%AD%A3%E5%88%99%E5%8C%96%E5%89%AA%E6%9E%9D)
* [LeNet](#lenet)
* [基于L1权重大小的剪枝](#%E5%9F%BA%E4%BA%8El1%E6%9D%83%E9%87%8D%E5%A4%A7%E5%B0%8F%E7%9A%84%E5%89%AA%E6%9E%9D)
* [基于L2权重大小的剪枝](#%E5%9F%BA%E4%BA%8El2%E6%9D%83%E9%87%8D%E5%A4%A7%E5%B0%8F%E7%9A%84%E5%89%AA%E6%9E%9D)
* [基于梯度大小](#%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E5%A4%A7%E5%B0%8F)
* [基于尺度](#%E5%9F%BA%E4%BA%8E%E5%B0%BA%E5%BA%A6)
* [基于二阶](#%E5%9F%BA%E4%BA%8E%E4%BA%8C%E9%98%B6)
* [剪枝频率](#%E5%89%AA%E6%9E%9D%E9%A2%91%E7%8E%87)
* [迭代剪枝](#%E8%BF%AD%E4%BB%A3%E5%89%AA%E6%9E%9D)
* [单次剪枝](#%E5%8D%95%E6%AC%A1%E5%89%AA%E6%9E%9D)
* [剪枝时机](#%E5%89%AA%E6%9E%9D%E6%97%B6%E6%9C%BA)
* [训练后剪枝](#%E8%AE%AD%E7%BB%83%E5%90%8E%E5%89%AA%E6%9E%9D)
* [训练时剪枝](#%E8%AE%AD%E7%BB%83%E6%97%B6%E5%89%AA%E6%9E%9D)
* [训练前剪枝](#%E8%AE%AD%E7%BB%83%E5%89%8D%E5%89%AA%E6%9E%9D)
* [剪枝时机总结](#%E5%89%AA%E6%9E%9D%E6%97%B6%E6%9C%BA%E6%80%BB%E7%BB%93)
* [剪枝比例](#%E5%89%AA%E6%9E%9D%E6%AF%94%E4%BE%8B)
* [代码](#%E4%BB%A3%E7%A0%81)

   \_\_EOF\_\_

       - **本文作者：** [PASSION](https://github.com)
 - **本文链接：** [https://github.com/passion2021/p/18548165](https://github.com)
 - **关于博主：** 评论和私信会在第一时间回复。或者[直接私信](https://github.com)我。
 - **版权声明：** 本博客所有文章除特别声明外，均采用 [BY\-NC\-SA](https://github.com "BY-NC-SA") 许可协议。转载请注明出处！
 - **声援博主：** 如果您觉得文章对您有帮助，可以点击文章右下角**【[推荐](javascript:void(0);)】**一下。
     
