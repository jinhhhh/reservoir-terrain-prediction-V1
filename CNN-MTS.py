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
import seaborn as sns

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 加载并预处理数据
def load_and_preprocess_data(cross_section_dir, feature_file, history_size=4, target_size=2):
    path_dir = Path(cross_section_dir)
    file_list = sorted(
        [f for f in path_dir.glob('*.xlsx')],
        key=lambda x: int(x.stem.replace('黄河', ''))
    )
    print(f"Found {len(file_list)} cross-section files.")

    all_data_list = []
    for file in file_list:
        df = pd.read_excel(file)
        df = df.drop(0)
        all_data_list.append(df.iloc[:, 1:].values)

    one_series = np.array(all_data_list)
    print(f"Raw time series data shape: {one_series.shape}")

    df_str = pd.read_excel(feature_file, header=1, index_col=0)
    original_str_data = df_str.values
    print(f"Structured feature data shape: {original_str_data.shape}")

    one_series_normalized = one_series

    data, str_data, labels = [], [], []
    num_years, num_spatial, num_cross = one_series_normalized.shape

    for year in range(num_years):
        for cross in range(num_cross):
            for spatial in range(num_spatial):
                if year >= history_size and year + target_size <= num_years:
                    history = one_series_normalized[year - history_size:year, spatial, cross]
                    data.append(history)

                    struct = original_str_data[year - history_size:year, :]
                    str_data.append(struct)

                    target = one_series[year:year + target_size, spatial, cross]
                    labels.append(target)

    data = np.array(data)
    str_data = np.array(str_data)
    labels = np.array(labels)

    print(f"Multi-variable data shape: {data.shape}, Structured data shape: {str_data.shape}, Labels shape: {labels.shape}")

    samples_per_year = num_spatial * num_cross

    train_years = 8
    val_years = 1
    test_years = 1

    train_samples = train_years * samples_per_year
    val_samples = val_years * samples_per_year
    test_samples = test_years * samples_per_year

    x_train = data[:train_samples]
    str_train = str_data[:train_samples]
    y_train = labels[:train_samples]

    x_val = data[train_samples:train_samples + val_samples]
    str_val = str_data[train_samples:train_samples + val_samples]
    y_val = labels[train_samples:train_samples + val_samples]

    x_test = data[train_samples + val_samples:train_samples + val_samples + test_samples]
    str_test = str_data[train_samples + val_samples:train_samples + val_samples + test_samples]
    y_test = labels[train_samples + val_samples:train_samples + val_samples + test_samples]

    print(f"Train set: {x_train.shape}, Validation set: {x_val.shape}, Test set: {x_test.shape}")

    return x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test

# 自定义 RAE 损失函数
class RAE_Loss(nn.Module):
    def __init__(self, y_baseline):
        super(RAE_Loss, self).__init__()
        self.register_buffer('y_baseline', torch.tensor(y_baseline, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        absolute_errors = torch.abs(y_true - y_pred).sum()
        baseline_errors = torch.abs(y_true - self.y_baseline).sum()
        return absolute_errors / (baseline_errors + 1e-10)

# 定义 RSE 计算函数
def relative_squared_error(y_true, y_pred, y_baseline):
    squared_errors = np.sum((y_true - y_pred) ** 2)
    baseline_squared_errors = np.sum((y_true - y_baseline) ** 2)
    return squared_errors / (baseline_squared_errors + 1e-10)

# 只使用CNN的模型
class CNN_Model(nn.Module):
    def __init__(self, history_size, structured_feature_size, future_target, num_features, lstm_hidden_size=128, lstm_num_layers=2):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv_str1 = nn.Conv1d(in_channels=structured_feature_size, out_channels=64, kernel_size=3, padding=1)
        self.bn_str1 = nn.BatchNorm1d(64)
        self.conv_str2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn_str2 = nn.BatchNorm1d(128)
        self.conv_str3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn_str3 = nn.BatchNorm1d(256)
        self.pool_str = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(256 * (history_size // 2) + 256 * (history_size // 2), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, future_target)

    def forward(self, x, b_str):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        b_str = b_str.permute(0, 2, 1)
        b_str = F.relu(self.bn_str1(self.conv_str1(b_str)))
        b_str = F.relu(self.bn_str2(self.conv_str2(b_str)))
        b_str = F.relu(self.bn_str3(self.conv_str3(b_str)))
        b_str = self.pool_str(b_str)
        b_str = b_str.view(b_str.size(0), -1)

        combined = torch.cat((x, b_str), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        out = self.fc3(combined)
        return out

# 保存预测结果到Excel
def save_predictions_to_excel(predictions, original_file, save_dir, target_years=[22,23]):
    original_df = pd.read_excel(original_file)
    original_df = original_df.drop(0)
    sections = original_df['section']

    num_cross = 56
    num_spatial = 100

    predictions = predictions.reshape(num_cross, num_spatial, len(target_years))

    columns = [f'黄河{56 - i}' for i in range(num_cross)]

    for year_idx, year in enumerate(target_years):
        df_pred = pd.DataFrame(columns=['section'] + columns)
        df_pred['section'] = sections

        for cross in range(num_cross):
            column_name = f'黄河{56 - cross}'
            df_pred[column_name] = predictions[cross, :, year_idx]

        save_path = os.path.join(save_dir, f'黄河{year}.xlsx')
        df_pred.to_excel(save_path, index=False)
        print(f"预测文件已保存到 {save_path}")

# 主函数
def main():
    cross_section_dir = "黄河断面插补Ⅱ - 深度学习 - 1"
    feature_file = "/tezhengzhi-2.xlsx"

    history_size = 4
    target_size = 2
    x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test = load_and_preprocess_data(
        cross_section_dir, feature_file, history_size, target_size
    )

    y_train_scaled = y_train
    y_val_scaled = y_val
    y_test_scaled = y_test

    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    y_baseline = y_train_scaled.mean(axis=0)
    print("y_baseline:", y_baseline)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(str_train, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(str_val, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(str_test, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32)
    )

    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    num_features = 1
    structured_feature_size = 12
    future_target = target_size
    model = CNN_Model(history_size, structured_feature_size, future_target, num_features).to(device)

    loss_fn = RAE_Loss(y_baseline).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        total_loss = 0
        for batch, (b_x, b_str, y) in enumerate(dataloader):
            b_x, b_str, y = b_x.to(device), b_str.to(device), y.to(device)
            pred = model(b_x, b_str)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

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

    epochs = 50
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            save_path = "/output data/CNN/best_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        else:
            trigger_times += 1
            print(f"Trigger Times: {trigger_times}")
            if trigger_times >= patience:
                print("Early Stopping!")
                break

    print("Training Completed!")

    plt.figure(figsize=(10,6))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss (RAE)')
    plt.plot(range(1,len(val_losses)+1), val_losses, label='Validation Loss (RAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    loss_curve_path = "/output data/CNN/figures/loss_curve.png"
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"损失曲线已保存到 {loss_curve_path}")

    plt.show()

    best_model_path = "/output data/CNN/best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for b_x, b_str, y in test_dl:
            b_x, b_str = b_x.to(device), b_str.to(device)
            pred = model(b_x, b_str).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    all_pred_inverse = all_pred
    all_true_inverse = all_true

    all_pred_inverse = np.maximum(all_pred_inverse, 0)
    all_pred_inverse[all_pred_inverse < 1] = 0

    print("Sample predictions after clamping:")
    print(all_pred_inverse[:5])

    print("Sample true values:")
    print(all_true_inverse[:5])

    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    mae = mean_absolute_error(all_true_inverse, all_pred_inverse)
    mse = mean_squared_error(all_true_inverse, all_pred_inverse)
    r2 = r2_score(all_true_inverse, all_pred_inverse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    rae = np.sum(np.abs(all_true_inverse - all_pred_inverse)) / np.sum(np.abs(all_true_inverse - y_baseline))
    rse = np.sum((all_true_inverse - all_pred_inverse) ** 2) / np.sum((all_true_inverse - y_baseline) ** 2)

    print(f"RAE: {rae:.4f}")
    print(f"RSE: {rse:.4f}")

    epsilon = 1e-10

    non_zero_mask = all_true_inverse != 0
    relative_errors = np.zeros_like(all_true_inverse)
    relative_errors[non_zero_mask] = np.abs(all_true_inverse[non_zero_mask] - all_pred_inverse[non_zero_mask]) / np.abs(all_true_inverse[non_zero_mask])

    relative_errors[~non_zero_mask] = 0

    relative_mae = np.mean(relative_errors[non_zero_mask])
    relative_median = np.median(relative_errors[non_zero_mask])
    relative_std = np.std(relative_errors[non_zero_mask])

    print(f"Relative MAE: {relative_mae * 100:.2f}%")
    print(f"Relative Median Error: {relative_median * 100:.2f}%")
    print(f"Relative Error Std Dev: {relative_std * 100:.2f}%")

    plt.figure(figsize=(10,6))
    sns.histplot(relative_errors[non_zero_mask], bins=50, kde=True)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative Errors')
    plt.show()

    print("Sample relative errors:")
    print(relative_errors[:5])

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
    metrics_save_path = "/output data/CNN/metric_results"
    os.makedirs(metrics_save_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(metrics_save_path, 'evaluation_metrics.csv'), index=False)
    print(f"评估指标已保存到 {os.path.join(metrics_save_path, 'evaluation_metrics.csv')}")

    # 保存预测结果
    predictions_save_path = "/output data/CNN/prediction_results"
    os.makedirs(predictions_save_path, exist_ok=True)

    for idx in range(target_size):
        pred_year = all_pred_inverse[:, idx]
        pred_df = pd.DataFrame(pred_year, columns=[f'Pred_Year{idx + 1}'])
        pred_df.to_csv(os.path.join(predictions_save_path, f'pred_result_year{idx + 1}.csv'), index=False)
        print(f"保存文件 {os.path.join(predictions_save_path, f'pred_result_year{idx + 1}.csv')} 成功")

    # 保存预测结果为 Excel 格式
    original_file = "/黄河断面插补Ⅱ - 深度学习 - 1/黄河09.xlsx"
    save_dir_excel = "/output data/CNN/prediction_results_excel"
    os.makedirs(save_dir_excel, exist_ok=True)
    save_predictions_to_excel(all_pred_inverse, original_file, save_dir_excel, target_years=[22, 23])

    print("所有任务完成！")


if __name__ == '__main__':
    main()
