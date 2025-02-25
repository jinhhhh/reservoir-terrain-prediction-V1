import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns  # 新增 seaborn 用于可视化

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(cross_section_dir, feature_file, history_size=4, target_size=2):
    """
    加载并预处理断面数据和结构化特征数据。

    参数:
    - cross_section_dir: 断面数据文件夹路径
    - feature_file: 结构化特征数据文件路径
    - history_size: 历史时间步长
    - target_size: 预测未来时间步长

    返回:
    - x_train, str_train, y_train: 训练集特征、结构化数据和标签
    - x_val, str_val, y_val: 验证集特征、结构化数据和标签
    - x_test, str_test, y_test: 测试集特征、结构化数据和标签
    """
    # 获取所有断面文件，按年份排序
    path_dir = Path(cross_section_dir)
    file_list = sorted(
        [f for f in path_dir.glob('*.xlsx')],
        key=lambda x: int(x.stem.replace('黄河', ''))
    )
    print(f"找到 {len(file_list)} 个断面文件。")

    all_data_list = []
    for file in file_list:
        df = pd.read_excel(file)
        df = df.drop(0)  # 删除第一行
        all_data_list.append(df.iloc[:, 1:].values)  # (100,56)

    # 堆叠所有数据：形状为 (15,100,56)
    one_series = np.array(all_data_list)  # (15,100,56)
    print(f"原始时间序列数据形状: {one_series.shape}")

    # 读取结构化特征数据
    df_str = pd.read_excel(feature_file, header=1, index_col=0)
    original_str_data = df_str.values  # (15,12)
    print(f"结构化特征数据形状: {original_str_data.shape}")

    # **移除输入特征的归一化**
    # scaler_X = MinMaxScaler()
    # one_series_reshaped = one_series.reshape(-1, one_series.shape[2])  # (1500,56)
    # one_series_normalized = scaler_X.fit_transform(one_series_reshaped).reshape(one_series.shape)  # (15,100,56)
    one_series_normalized = one_series  # 使用原始数据

    # 生成样本
    data, str_data, labels = [], [], []
    num_years, num_spatial, num_cross = one_series_normalized.shape  # (15,100,56)

    for year in range(num_years):
        for cross in range(num_cross):
            for spatial in range(num_spatial):
                # 确保有足够的历史和目标时间步
                if year >= history_size and year + target_size <= num_years:
                    history = one_series_normalized[year - history_size:year, spatial, cross]  # (4,)
                    data.append(history)

                    # 对应的结构化特征
                    struct = original_str_data[year - history_size:year, :]  # (4,12)
                    str_data.append(struct)

                    # **目标数据应来自原始未归一化的数据**
                    target = one_series[year:year + target_size, spatial, cross]  # (2,)
                    labels.append(target)

    data = np.array(data)  # (N,4)
    str_data = np.array(str_data)  # (N,4,12)
    labels = np.array(labels)  # (N,2)

    print(f"多变量数据形状: {data.shape}, 结构化数据形状: {str_data.shape}, 标签形状: {labels.shape}")

    # 按年份划分数据集
    # history_size=4, target_size=2
    # year=4..13 可以生成样本（2009+4=2013年 到 2009+13=2022年）

    # 训练集：2009-2020年（year=0-11）
    # 验证集：2021年（year=12）
    # 测试集：2022年（year=13-14）

    # 计算样本数量
    samples_per_year = num_spatial * num_cross  # 100 * 56 = 5600

    # 训练集样本数量：8年 * 5600 = 44800
    # 验证集样本数量：1年 * 5600 = 5600
    # 测试集样本数量：1年 * 5600 = 5600

    train_years = 8  # 2009-2020
    val_years = 1     # 2021
    test_years = 1    # 2022

    train_samples = train_years * samples_per_year  # 8 * 5600 = 44800
    val_samples = val_years * samples_per_year      # 1 * 5600 = 5600
    test_samples = test_years * samples_per_year    # 1 * 5600 = 5600

    x_train = data[:train_samples]  # (44800,4)
    str_train = str_data[:train_samples]  # (44800,4,12)
    y_train = labels[:train_samples]  # (44800,2)

    x_val = data[train_samples:train_samples + val_samples]  # (5600,4)
    str_val = str_data[train_samples:train_samples + val_samples]  # (5600,4,12)
    y_val = labels[train_samples:train_samples + val_samples]  # (5600,2)

    x_test = data[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600,4)
    str_test = str_data[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600,4,12)
    y_test = labels[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600,2)

    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}, 测试集: {x_test.shape}")

    return x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test

# 自定义 RAE 损失函数
class RAE_Loss(nn.Module):
    def __init__(self, y_baseline):
        """
        初始化 RAE 损失函数。

        参数:
        - y_baseline: 基准值，通常是训练集标签的均值
        """
        super(RAE_Loss, self).__init__()
        # 将 y_baseline 转换为 tensor 并存储为 buffer
        self.register_buffer('y_baseline', torch.tensor(y_baseline, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        """
        前向传播计算 RAE 损失。

        参数:
        - y_pred: 预测值，形状为 (batch_size, target_size)
        - y_true: 真实值，形状为 (batch_size, target_size)

        返回:
        - RAE 损失
        """
        absolute_errors = torch.abs(y_true - y_pred).sum()
        baseline_errors = torch.abs(y_true - self.y_baseline).sum()
        return absolute_errors / (baseline_errors + 1e-10)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码。

        参数:
        - d_model: 特征维度
        - max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        添加位置编码。

        参数:
        - x: 输入，形状为 (seq_len, batch_size, d_model)

        返回:
        - 加上位置编码后的输出
        """
        x = x + self.pe[:x.size(0), :]
        return x

# CNN+Transformer的模型
class CNN_Transformer_Model(nn.Module):
    def __init__(self, history_size, structured_feature_size, future_target, num_features,
                 transformer_dim=128, transformer_heads=4, transformer_layers=2, transformer_dropout=0.1):
        super(CNN_Transformer_Model, self).__init__()
        # CNN部分处理时间序列数据
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 将序列长度缩减

        # Transformer部分处理结构化特征
        self.transformer_input_dim = structured_feature_size
        self.transformer_dim = transformer_dim
        self.embedding = nn.Linear(structured_feature_size, transformer_dim)  # 线性变换到Transformer维度
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=history_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads,
                                                    dropout=transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_layers)
        self.transformer_fc = nn.Linear(transformer_dim, 128)  # 全连接层处理Transformer输出

        # 融合部分
        # CNN输出：256 * (history_size//2)
        # Transformer输出：128
        self.fc1 = nn.Linear(256 * (history_size // 2) + 128, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, future_target)

    def forward(self, x, b_str):
        """
        前向传播函数。

        参数:
        - x: 时间序列数据，形状为 (batch_size, 1, history_size)
        - b_str: 结构化特征数据，形状为 (batch_size, history_size, structured_feature_size)

        返回:
        - 预测值，形状为 (batch_size, future_target)
        """
        # CNN处理时间序列数据
        x = F.relu(self.bn1(self.conv1(x)))  # (batch,64,history_size)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch,128,history_size)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch,256,history_size)
        x = self.pool(x)  # (batch,256,history_size//2)

        x = x.view(x.size(0), -1)  # (batch,256*(history_size//2))

        # Transformer处理结构化特征
        # b_str: (batch, history_size, structured_feature_size)
        b_str = self.embedding(b_str)  # (batch, history_size, transformer_dim)
        b_str = b_str.permute(1, 0, 2)  # 转换为 (history_size, batch, transformer_dim)
        b_str = self.pos_encoder(b_str)  # 添加位置编码
        transformer_out = self.transformer_encoder(b_str)  # (history_size, batch, transformer_dim)
        transformer_out = transformer_out.mean(dim=0)  # (batch, transformer_dim)
        transformer_out = F.relu(self.transformer_fc(transformer_out))  # (batch,128)

        # 融合CNN和Transformer的输出
        combined = torch.cat((x, transformer_out), dim=1)  # (batch,256*(history_size//2)+128)

        # 全连接层进行最终预测
        combined = F.relu(self.fc1(combined))  # (batch,512)
        combined = F.relu(self.fc2(combined))  # (batch,256)
        out = self.fc3(combined)  # (batch,future_target)

        return out

def save_predictions_to_excel(predictions, original_file, save_dir, target_years=[22,23]):
    """
    将预测结果保存为与原始Excel文件相同格式的文件。

    参数:
    - predictions: np.array, 形状为 (5600, 2)
    - original_file: str, 原始Excel文件路径，用于获取 'section' 列
    - save_dir: str, 保存预测文件的目录
    - target_years: list, 预测的年份列表，例如 [22,23]
    """
    # 读取原始文件以获取 'section' 列
    original_df = pd.read_excel(original_file)
    original_df = original_df.drop(0)  # 删除第一行
    sections = original_df['section']

    num_cross = 56
    num_spatial = 100

    # 重塑预测结果为 (56, 100, 2)
    predictions = predictions.reshape(num_cross, num_spatial, len(target_years))  # (56,100,2)

    # 创建列名称 '黄河56' 到 '黄河1'
    columns = [f'黄河{56 - i}' for i in range(num_cross)]  # ['黄河56', '黄河55', ..., '黄河1']

    for year_idx, year in enumerate(target_years):
        df_pred = pd.DataFrame(columns=['section'] + columns)
        df_pred['section'] = sections

        for cross in range(num_cross):
            column_name = f'黄河{56 - cross}'
            df_pred[column_name] = predictions[cross, :, year_idx]

        # 保存为Excel文件
        save_path = os.path.join(save_dir, f'黄河{year}.xlsx')
        df_pred.to_excel(save_path, index=False)
        print(f"预测文件已保存到 {save_path}")

# 主函数
def main():
    # 数据路径
    cross_section_dir = "/content/黄河断面插补Ⅱ - 深度学习 - 1"
    feature_file = "/content/tezhengzhi-2.xlsx"

    # 加载并预处理数据
    history_size = 4
    target_size = 2  # 保持为2
    x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test = load_and_preprocess_data(
        cross_section_dir, feature_file, history_size, target_size
    )

    # **移除标签的归一化**
    # scaler_y = MinMaxScaler()
    # y_train_scaled = scaler_y.fit_transform(y_train)
    # print("scaler_y min:", scaler_y.min_)
    # print("scaler_y scale:", scaler_y.scale_)
    # y_val_scaled = scaler_y.transform(y_val)
    # y_test_scaled = scaler_y.transform(y_test)

    # 使用原始标签
    y_train_scaled = y_train
    y_val_scaled = y_val
    y_test_scaled = y_test

    # **打印 y_train 的最小值和最大值以验证**
    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    # 计算基准值（y_baseline）为训练集标签的均值
    y_baseline = y_train_scaled.mean(axis=0)
    print("y_baseline:", y_baseline)

    # 创建TensorDataset
    # x: (samples,4) -> (samples,1,4)
    # str_data: (samples,4,12)
    # y: (samples,2)
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).unsqueeze(1),  # (samples,1,4)
        torch.tensor(str_train, dtype=torch.float32),            # (samples,4,12)
        torch.tensor(y_train_scaled, dtype=torch.float32)        # (samples,2)
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32).unsqueeze(1),    # (samples,1,4)
        torch.tensor(str_val, dtype=torch.float32),              # (samples,4,12)
        torch.tensor(y_val_scaled, dtype=torch.float32)          # (samples,2)
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32).unsqueeze(1),   # (samples,1,4)
        torch.tensor(str_test, dtype=torch.float32),             # (samples,4,12)
        torch.tensor(y_test_scaled, dtype=torch.float32)         # (samples,2)
    )

    # 创建DataLoader
    batch_size = 64  # 根据内存情况调整
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)  # 修改num_workers=2
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    num_features = 1  # 因为每个样本的时间序列特征数量为1
    structured_feature_size = 12  # 从结构化特征数据
    future_target = target_size
    model = CNN_Transformer_Model(history_size, structured_feature_size, future_target, num_features).to(device)

    # 定义损失函数与优化器
    loss_fn = RAE_Loss(y_baseline).to(device)  # 修改为 RAE 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练函数
    def train_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        total_loss = 0
        for batch, (b_x, b_str, y) in enumerate(dataloader):
            b_x, b_str, y = b_x.to(device), b_str.to(device), y.to(device)

            # 预测
            pred = model(b_x, b_str)

            # 计算损失
            loss = loss_fn(pred, y)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    # 评估函数
    def evaluate(dataloader, model, loss_fn):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for b_x, b_str, y in dataloader:
                b_x, b_str, y = b_x.to(device), b_str.to(device), y.to(device)
                pred = model(b_x, b_str)
                loss = loss_fn(pred, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # 训练过程
    epochs = 50  # 您可以根据需要手动调整这个值
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(epochs):
        train_loss = train_epoch(train_dl, model, loss_fn, optimizer)
        val_loss = evaluate(val_dl, model, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train RAE Loss = {train_loss:.6f}, Val RAE Loss = {val_loss:.6f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # 保存最佳模型
            save_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/best_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 为避免FutureWarning，明确指定加载方式
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        else:
            trigger_times += 1
            print(f"Trigger Times: {trigger_times}")
            if trigger_times >= patience:
                print("Early Stopping!")
                break

    print("Training Completed!")

    # 可视化损失曲线
    plt.figure(figsize=(10,6))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss (RAE)')
    plt.plot(range(1,len(val_losses)+1), val_losses, label='Validation Loss (RAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像
    loss_curve_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/figures/loss_curve.png"
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"损失曲线已保存到 {loss_curve_path}")

    plt.show()

    # 加载最佳模型
    best_model_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"已加载最佳模型从 {best_model_path}")

    # 评估测试集
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for b_x, b_str, y in test_dl:
            b_x, b_str = b_x.to(device), b_str.to(device)
            pred = model(b_x, b_str).cpu().numpy()  # (batch,2)
            all_pred.append(pred)  # list of (batch,2)
            all_true.append(y.cpu().numpy())  # list of (batch,2)
    all_pred = np.concatenate(all_pred, axis=0)  # (5600,2)
    all_true = np.concatenate(all_true, axis=0)  # (5600,2)

    # **移除逆归一化**
    # all_pred_inverse = scaler_y.inverse_transform(all_pred)  # (5600,2)
    # all_true_inverse = scaler_y.inverse_transform(all_true)  # (5600,2)
    all_pred_inverse = all_pred  # 使用原始预测值
    all_true_inverse = all_true  # 使用原始真实值

    # 将所有负值设为0（根据实际情况决定是否需要）
    all_pred_inverse = np.maximum(all_pred_inverse, 0)

    # **新增** 将所有预测值小于1的设为0
    all_pred_inverse[all_pred_inverse < 1] = 0

    # 添加调试输出
    print("样本预测值（截断后）:")
    print(all_pred_inverse[:5])

    print("样本真实值:")
    print(all_true_inverse[:5])

    # **打印 y_train 的最小值和最大值以验证**
    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    # 计算评估指标
    mae = mean_absolute_error(all_true_inverse, all_pred_inverse)
    mse = mean_squared_error(all_true_inverse, all_pred_inverse)
    r2 = r2_score(all_true_inverse, all_pred_inverse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    # 计算相对绝对误差（RAE）和相对均方误差（RSE）
    rae = np.sum(np.abs(all_true_inverse - all_pred_inverse)) / np.sum(np.abs(all_true_inverse - y_baseline))
    rse = np.sum((all_true_inverse - all_pred_inverse) ** 2) / np.sum((all_true_inverse - y_baseline) ** 2)

    print(f"RAE: {rae:.4f}")
    print(f"RSE: {rse:.4f}")

    # 计算每个样本的相对误差
    epsilon = 1e-10  # 防止除以零

    # 仅在真实值非零时计算相对误差
    non_zero_mask = all_true_inverse != 0
    relative_errors = np.zeros_like(all_true_inverse)
    relative_errors[non_zero_mask] = np.abs(all_true_inverse[non_zero_mask] - all_pred_inverse[non_zero_mask]) / np.abs(all_true_inverse[non_zero_mask])

    # 将真实值为零的相对误差设为0或其他适当的值
    relative_errors[~non_zero_mask] = 0  # 或设为某个大值，如 1

    # 计算相对误差的统计指标
    relative_mae = np.mean(relative_errors[non_zero_mask])
    relative_median = np.median(relative_errors[non_zero_mask])
    relative_std = np.std(relative_errors[non_zero_mask])

    print(f"相对 MAE: {relative_mae * 100:.2f}%")
    print(f"相对中位误差: {relative_median * 100:.2f}%")
    print(f"相对误差标准差: {relative_std * 100:.2f}%")

    # 绘制相对误差的分布图
    plt.figure(figsize=(10,6))
    sns.histplot(relative_errors[non_zero_mask], bins=50, kde=True)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative Errors')
    plt.show()

    # **打印相对误差的一些样本**
    print("样本相对误差:")
    print(relative_errors[:5])

    # 保存评估指标
    metrics = {
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2],
        'RAE': [rae],
        'RSE': [rse],
        'Relative MAE (%)': [relative_mae * 100],
        'Relative Median Error (%)': [relative_median * 100],
        'Relative Error Std Dev (%)': [relative_std * 100]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_save_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/metric_results"
    os.makedirs(metrics_save_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_save_path, 'evaluation_metrics.csv'), index=False)
    print(f"评估指标已保存到 {os.path.join(metrics_save_path, 'evaluation_metrics.csv')}")

    # 保存预测结果
    predictions_save_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/prediction_results"
    os.makedirs(predictions_save_path, exist_ok=True)

    for idx in range(target_size):
        pred_year = all_pred_inverse[:, idx]
        pred_df = pd.DataFrame(pred_year, columns=[f'Pred_Year{idx+1}'])
        pred_df.to_csv(os.path.join(predictions_save_path, f'pred_result_year{idx+1}.csv'), index=False)
        print(f"保存文件 {os.path.join(predictions_save_path, f'pred_result_year{idx+1}.csv')} 成功")

    # 保存预测结果为 Excel 格式
    original_file = "/content/黄河断面插补Ⅱ - 深度学习 - 1/黄河09.xlsx"  # 用于获取 'section' 列
    save_dir_excel = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/prediction_results_excel"
    os.makedirs(save_dir_excel, exist_ok=True)
    save_predictions_to_excel(all_pred_inverse, original_file, save_dir_excel, target_years=[22,23])

    print("所有任务完成！")

if __name__ == '__main__':
    main()
