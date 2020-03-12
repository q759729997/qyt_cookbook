# 数据读取与处理

## 读取数据

- PyTorch提供了`data`包来读取数据。

~~~python
features = torch.zeros(20, 2)  # 第一个维度表示样本数目，切分batch时以此为依据
print('features:', features.shape)  # features: torch.Size([20, 2])
labels = torch.ones(features.shape[0])  # labels: torch.Size([20])
print('labels:', labels.shape)  # data_iter len: 3
batch_size = 3
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)  # tensors that have the same size of the first dimension.
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
print('data_iter len:', len(data_iter))  # data_iter len: 7
for X, y in data_iter:
print(X, y)
break
"""输出
tensor([[0., 0.],
[0., 0.],
[0., 0.]]) tensor([1., 1., 1.])
"""
~~~

- `Data.TensorDataset`：输入的tensors需要**第一维的大小一致**.
- `Data.DataLoader`：重要参数：

~~~wiki
batch_size(default: 1):批次大小;
shuffle(default: False):每个epoch取数据时是否重新打乱数据;
drop_last(default: False):不满batch_size时，最后一个批次是否删除;
num_workers:(default: 0):多线程处理数据,windows下暂时不能设置多线程;
~~~

