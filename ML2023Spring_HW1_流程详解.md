# ML2023Spring_HW1.ipynb 操作流程与方法详解

本作业是一个 **COVID-19 确诊数回归预测** 任务，目标是用 PyTorch 建立 DNN 回归模型，输出测试集每一行样本的 `tested_positive` 预测值。

## 1. 环境与数据准备

1. 在 Colab 挂载 Google Drive（可选，用于读写数据与模型）。
2. 用 `nvidia-smi` 检查 GPU。
3. 用 `gdown` 下载 `covid_train.csv` 与 `covid_test.csv`（也可用 Kaggle/Dropbox 链接）。

## 2. 导入套件

核心套件：
- 数值与数据处理：`numpy`, `pandas`
- 深度学习：`torch`, `torch.nn`, `torch.utils.data`
- 日志与可视化：`tensorboard` / `SummaryWriter`
- 训练过程可视化：`tqdm`

## 3. 工具函数（Utility）

### `same_seed(seed)`
作用：固定随机种子，提升实验可复现性。
- 关闭 cuDNN benchmark 的随机优化路径
- 固定 numpy 与 torch（CPU/GPU）随机状态

### `train_valid_split(data_set, valid_ratio, seed)`
作用：将训练数据按比例切成 train/valid。
- 验证集大小 = `int(valid_ratio * len(data_set))`
- 通过 `random_split(..., generator=torch.Generator().manual_seed(seed))` 保证切分可复现

### `predict(test_loader, model, device)`
作用：批量推理测试集。
- `model.eval()` 切到推理模式
- `torch.no_grad()` 关闭梯度
- 拼接每个 batch 结果并输出 `numpy`

## 4. 数据集封装：`COVID19Dataset`

这个类继承 `Dataset`，统一训练与推理的数据接口：
- 训练/验证时：返回 `(x[idx], y[idx])`
- 测试时（无标签）：只返回 `x[idx]`

这样可直接被 `DataLoader` 批量加载。

## 5. 模型结构：`My_Model`

当前提供的是一个三层全连接网络：
- `Linear(input_dim, 16)` + `ReLU`
- `Linear(16, 8)` + `ReLU`
- `Linear(8, 1)`

`forward` 最后用 `squeeze(1)` 把形状从 `(B,1)` 变成 `(B,)`，与回归标签形状对齐。

> 作业鼓励你改这块（如层数、隐藏维度、激活函数、正则化等）。

## 6. 特征选择：`select_feat(...)`

输入：`train_data, valid_data, test_data`
- 训练/验证集最后一列视为标签 `y`
- 其余列是特征 `x`

逻辑：
- `select_all=True`：使用全部特征
- `False`：示例只取前 5 列（应自行改成更合理的特征子集）

这是作业中影响成绩的重要调参点之一。

## 7. 训练流程：`trainer(...)`

### 损失函数
- `nn.MSELoss(reduction='mean')`：标准回归损失

### 优化器
- 当前是 `SGD(lr, momentum=0.7)`
- 注释中提示可改成其他优化器，并可加入 L2（`weight_decay`）

### 每轮训练（epoch）
1. `model.train()`
2. 遍历 `train_loader`：
   - `zero_grad()`
   - 前向：`pred = model(x)`
   - 算损失：`loss = criterion(pred, y)`
   - 反向：`loss.backward()`
   - 更新：`optimizer.step()`
3. 记录并打印训练损失

### 每轮验证
1. `model.eval()`
2. `torch.no_grad()` 下遍历 `valid_loader`
3. 计算平均验证损失

### 早停与存模
- 若验证损失刷新最优：
  - 保存到 `config['save_path']`
  - 早停计数归零
- 否则早停计数 +1
- 达到 `early_stop` 后提前终止训练

## 8. 超参数配置：`config`

关键参数含义：
- `seed`：随机种子
- `valid_ratio`：验证集比例
- `n_epochs`：最大训练轮次
- `batch_size`：批量大小
- `learning_rate`：学习率
- `early_stop`：连续多少轮不提升就停止
- `save_path`：最佳模型保存路径

> 这份基线设置里 `learning_rate=1e-5` 且 `n_epochs=5000`，训练会比较慢，通常需要联合调整。

## 9. 数据加载与开训

1. 读入 CSV
2. 切分 train/valid
3. 按配置选择特征
4. 构建三个 `COVID19Dataset`
5. 构建 `DataLoader`（训练集启用 `shuffle=True`）
6. 实例化模型并调用 `trainer(...)`

## 10. 推理与输出提交文件

1. 用同样结构重建模型
2. `load_state_dict` 载入最佳参数
3. `predict(...)` 得到测试集预测
4. `save_pred(...)` 写出 `pred.csv`，列名是 `id,tested_positive`

最后可在 Colab 中 `files.download('pred.csv')` 下载提交。

## 11. 建议的改进方向（拿分关键）

1. **特征工程**：手动筛特征、标准化、交互特征。
2. **模型结构**：更深网络、BatchNorm、Dropout。
3. **优化策略**：改 Adam/AdamW、学习率调度器。
4. **正则化**：`weight_decay`、早停阈值优化。
5. **验证策略**：固定随机种子，多次重跑取稳健结果。

---

一句话总结：
这个 notebook 的完整流水线是 **数据读取 → 切分/选特征 → Dataset/DataLoader → DNN 回归训练（MSE + 早停）→ 载入最优模型推理 → 导出 `pred.csv`**。
