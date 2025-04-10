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
import seaborn as sns  # Import seaborn for visualization

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(cross_section_dir, feature_file, history_size=4, target_size=2):
    """
    Load and preprocess cross-sectional data and structured feature data.

    Parameters:
    - cross_section_dir: Path to the directory containing cross-sectional data files
    - feature_file: Path to the structured feature data file
    - history_size: Number of historical time steps
    - target_size: Number of future time steps to predict

    Returns:
    - x_train, str_train, y_train: Training features, structured data, and labels
    - x_val, str_val, y_val: Validation features, structured data, and labels
    - x_test, str_test, y_test: Test features, structured data, and labels
    """
    # Get all cross-sectional files and sort by year
    path_dir = Path(cross_section_dir)
    file_list = sorted(
        [f for f in path_dir.glob('*.xlsx')],
        key=lambda x: int(x.stem.replace('Yellow River', ''))
    )
    print(f"Found {len(file_list)} cross-sectional files.")

    all_data_list = []
    for file in file_list:
        df = pd.read_excel(file)
        df = df.drop(0)  # Remove the first row
        all_data_list.append(df.iloc[:, 1:].values)  # (100, 56)

    # Stack all data: shape (15, 100, 56)
    one_series = np.array(all_data_list)
    print(f"Raw time series data shape: {one_series.shape}")

    # Load structured feature data
    df_str = pd.read_excel(feature_file, header=1, index_col=0)
    original_str_data = df_str.values  # (15, 12)
    print(f"Structured feature data shape: {original_str_data.shape}")

    # No normalization for input features
    one_series_normalized = one_series

    # Generate samples
    data, str_data, labels = [], [], []
    num_years, num_spatial, num_cross = one_series_normalized.shape  # (15, 100, 56)

    for year in range(num_years):
        for cross in range(num_cross):
            for spatial in range(num_spatial):
                if year >= history_size and year + target_size <= num_years:
                    history = one_series_normalized[year - history_size:year, spatial, cross]  # (4,)
                    data.append(history)

                    struct = original_str_data[year - history_size:year, :]  # (4, 12)
                    str_data.append(struct)

                    target = one_series[year:year + target_size, spatial, cross]  # (2,)
                    labels.append(target)

    data = np.array(data)  # (N, 4)
    str_data = np.array(str_data)  # (N, 4, 12)
    labels = np.array(labels)  # (N, 2)

    print(f"Multivariate data shape: {data.shape}, Structured data shape: {str_data.shape}, Labels shape: {labels.shape}")

    # Dataset split by years
    samples_per_year = num_spatial * num_cross  # 100 * 56 = 5600

    train_years = 8  # 2009-2020
    val_years = 1     # 2021
    test_years = 1    # 2022

    train_samples = train_years * samples_per_year  # 8 * 5600 = 44800
    val_samples = val_years * samples_per_year      # 1 * 5600 = 5600
    test_samples = test_years * samples_per_year    # 1 * 5600 = 5600

    x_train = data[:train_samples]  # (44800, 4)
    str_train = str_data[:train_samples]  # (44800, 4, 12)
    y_train = labels[:train_samples]  # (44800, 2)

    x_val = data[train_samples:train_samples + val_samples]  # (5600, 4)
    str_val = str_data[train_samples:train_samples + val_samples]  # (5600, 4, 12)
    y_val = labels[train_samples:train_samples + val_samples]  # (5600, 2)

    x_test = data[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600, 4)
    str_test = str_data[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600, 4, 12)
    y_test = labels[train_samples + val_samples:train_samples + val_samples + test_samples]  # (5600, 2)

    print(f"Training set: {x_train.shape}, Validation set: {x_val.shape}, Test set: {x_test.shape}")

    return x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test

# Custom RAE Loss Function
class RAE_Loss(nn.Module):
    def __init__(self, y_baseline):
        """
        Initialize the RAE loss function.

        Parameters:
        - y_baseline: Baseline value, typically the mean of training labels
        """
        super(RAE_Loss, self).__init__()
        self.register_buffer('y_baseline', torch.tensor(y_baseline, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        """
        Forward pass to compute RAE loss.

        Parameters:
        - y_pred: Predictions, shape (batch_size, target_size)
        - y_true: Ground truth, shape (batch_size, target_size)

        Returns:
        - RAE loss value
        """
        absolute_errors = torch.abs(y_true - y_pred).sum()
        baseline_errors = torch.abs(y_true - self.y_baseline).sum()
        return absolute_errors / (baseline_errors + 1e-10)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Positional encoding for sequence data.

        Parameters:
        - d_model: Dimension of the model
        - max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input sequences.

        Parameters:
        - x: Input sequence, shape (seq_len, batch_size, d_model)

        Returns:
        - Output with positional encoding, shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x

# CNN-Transformer Model
class CNN_Transformer_Model(nn.Module):
    def __init__(self, history_size, structured_feature_size, future_target, num_features,
                 transformer_dim=128, transformer_heads=4, transformer_layers=2, transformer_dropout=0.1):
        super(CNN_Transformer_Model, self).__init__()
        # CNN for time series data
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduce sequence length

        # Transformer for structured features
        self.transformer_input_dim = structured_feature_size
        self.transformer_dim = transformer_dim
        self.embedding = nn.Linear(structured_feature_size, transformer_dim)  # Project to transformer dimension
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=history_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=transformer_heads,
                                                    dropout=transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_layers)
        self.transformer_fc = nn.Linear(transformer_dim, 128)  # Fully connected for transformer output

        # Fusion layers
        self.fc1 = nn.Linear(256 * (history_size // 2) + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, future_target)

    def forward(self, x, b_str):
        """
        Forward pass of the model.

        Parameters:
        - x: Time series data, shape (batch_size, 1, history_size)
        - b_str: Structured feature data, shape (batch_size, history_size, structured_feature_size)

        Returns:
        - Predictions, shape (batch_size, future_target)
        """
        # CNN processing
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, history_size)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, history_size)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 256, history_size)
        x = self.pool(x)  # (batch, 256, history_size//2)
        x = x.view(x.size(0), -1)  # (batch, 256*(history_size//2))

        # Transformer processing
        b_str = self.embedding(b_str)  # (batch, history_size, transformer_dim)
        b_str = b_str.permute(1, 0, 2)  # (history_size, batch, transformer_dim)
        b_str = self.pos_encoder(b_str)  # Add positional encoding
        transformer_out = self.transformer_encoder(b_str)  # (history_size, batch, transformer_dim)
        transformer_out = transformer_out.mean(dim=0)  # (batch, transformer_dim)
        transformer_out = F.relu(self.transformer_fc(transformer_out))  # (batch, 128)

        # Fusion
        combined = torch.cat((x, transformer_out), dim=1)  # (batch, 256*(history_size//2)+128)

        # Final prediction layers
        combined = F.relu(self.fc1(combined))  # (batch, 512)
        combined = F.relu(self.fc2(combined))  # (batch, 256)
        out = self.fc3(combined)  # (batch, future_target)

        return out

def save_predictions_to_excel(predictions, original_file, save_dir, target_years=[22,23]):
    """
    Save prediction results to Excel files in the format of the original data.

    Parameters:
    - predictions: np.array, shape (5600, 2)
    - original_file: str, path to the original Excel file for 'section' column
    - save_dir: str, directory to save prediction files
    - target_years: list, years to predict (e.g., [22,23])
    """
    # Load original file to get 'section' column
    original_df = pd.read_excel(original_file)
    original_df = original_df.drop(0)  # Remove the first row
    sections = original_df['section']

    num_cross = 56
    num_spatial = 100

    # Reshape predictions to (56, 100, 2)
    predictions = predictions.reshape(num_cross, num_spatial, len(target_years))  # (56, 100, 2)

    # Create column names 'Yellow River56' to 'Yellow River1'
    columns = [f'Yellow River{56 - i}' for i in range(num_cross)]  # ['Yellow River56', 'Yellow River55', ..., 'Yellow River1']

    for year_idx, year in enumerate(target_years):
        df_pred = pd.DataFrame(columns=['section'] + columns)
        df_pred['section'] = sections

        for cross in range(num_cross):
            column_name = f'Yellow River{56 - cross}'
            df_pred[column_name] = predictions[cross, :, year_idx]

        # Save to Excel
        save_path = os.path.join(save_dir, f'Yellow River{year}.xlsx')
        df_pred.to_excel(save_path, index=False)
        print(f"Prediction file saved to {save_path}")

# Main function
def main():
    # Data paths
    cross_section_dir = "/content/Yellow River section interpolationⅡ - Deep Learning - 1"
    feature_file = "/content/tezhengzhi-2.xlsx"

    # Load and preprocess data
    history_size = 4
    target_size = 2
    x_train, str_train, y_train, x_val, str_val, y_val, x_test, str_test, y_test = load_and_preprocess_data(
        cross_section_dir, feature_file, history_size, target_size
    )

    # Use original labels without normalization
    y_train_scaled = y_train
    y_val_scaled = y_val
    y_test_scaled = y_test

    # Print min and max of y_train for verification
    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    # Calculate baseline as mean of training labels
    y_baseline = y_train_scaled.mean(axis=0)
    print("y_baseline:", y_baseline)

    # Create TensorDataset
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32).unsqueeze(1),  # (samples, 1, 4)
        torch.tensor(str_train, dtype=torch.float32),            # (samples, 4, 12)
        torch.tensor(y_train_scaled, dtype=torch.float32)        # (samples, 2)
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32).unsqueeze(1),    # (samples, 1, 4)
        torch.tensor(str_val, dtype=torch.float32),              # (samples, 4, 12)
        torch.tensor(y_val_scaled, dtype=torch.float32)          # (samples, 2)
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32).unsqueeze(1),   # (samples, 1, 4)
        torch.tensor(str_test, dtype=torch.float32),             # (samples, 4, 12)
        torch.tensor(y_test_scaled, dtype=torch.float32)         # (samples, 2)
    )

    # Create DataLoader
    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model
    num_features = 1
    structured_feature_size = 12
    future_target = target_size
    model = CNN_Transformer_Model(history_size, structured_feature_size, future_target, num_features).to(device)

    # Loss function and optimizer
    loss_fn = RAE_Loss(y_baseline).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training epoch function
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

    # Evaluation function
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

    # Training loop with early stopping
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
            save_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/best_model.pth"
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

    # Plot loss curves
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss (RAE)')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss (RAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save loss curve
    loss_curve_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/figures/loss_curve.png"
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path, bbox_inches='tight', dpi=300)
    print(f"Loss curve saved to {loss_curve_path}")

    plt.show()

    # Load best model
    best_model_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/best_model.pth"
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    # Evaluate on test set
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for b_x, b_str, y in test_dl:
            b_x, b_str = b_x.to(device), b_str.to(device)
            pred = model(b_x, b_str).cpu().numpy()  # (batch, 2)
            all_pred.append(pred)
            all_true.append(y.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)  # (5600, 2)
    all_true = np.concatenate(all_true, axis=0)  # (5600, 2)

    # Apply clamping: set negative predictions to 0 and values <1 to 0
    all_pred_inverse = np.maximum(all_pred, 0)
    all_pred_inverse[all_pred_inverse < 1] = 0

    # Debugging outputs
    print("Sample predictions after clamping:")
    print(all_pred_inverse[:5])

    print("Sample true values:")
    print(all_true_inverse[:5])

    # Print y_train statistics
    print("y_train min:", y_train.min())
    print("y_train max:", y_train.max())

    # Calculate evaluation metrics
    mae = mean_absolute_error(all_true_inverse, all_pred_inverse)
    mse = mean_squared_error(all_true_inverse, all_pred_inverse)
    r2 = r2_score(all_true_inverse, all_pred_inverse)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    # Calculate RAE and RSE
    rae = np.sum(np.abs(all_true_inverse - all_pred_inverse)) / np.sum(np.abs(all_true_inverse - y_baseline))
    rse = np.sum((all_true_inverse - all_pred_inverse) ** 2) / np.sum((all_true_inverse - y_baseline) ** 2)

    print(f"RAE: {rae:.4f}")
    print(f"RSE: {rse:.4f}")

    # Calculate relative errors
    epsilon = 1e-10
    non_zero_mask = all_true_inverse != 0
    relative_errors = np.zeros_like(all_true_inverse)
    relative_errors[non_zero_mask] = np.abs(all_true_inverse[non_zero_mask] - all_pred_inverse[non_zero_mask]) / np.abs(all_true_inverse[non_zero_mask])
    relative_errors[~non_zero_mask] = 0

    # Statistical metrics for relative errors
    relative_mae = np.mean(relative_errors[non_zero_mask])
    relative_median = np.median(relative_errors[non_zero_mask])
    relative_std = np.std(relative_errors[non_zero_mask])

    print(f"Relative MAE: {relative_mae * 100:.2f}%")
    print(f"Relative Median Error: {relative_median * 100:.2f}%")
    print(f"Relative Error Std Dev: {relative_std * 100:.2f}%")

    # Plot relative error distribution
    plt.figure(figsize=(10,6))
    sns.histplot(relative_errors[non_zero_mask], bins=50, kde=True)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative Errors')
    plt.show()

    # Print sample relative errors
    print("Sample relative errors:")
    print(relative_errors[:5])

    # Save evaluation metrics
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
    print(f"Evaluation metrics saved to {os.path.join(metrics_save_path, 'evaluation_metrics.csv')}")

    # Save prediction results
    predictions_save_path = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/prediction_results"
    os.makedirs(predictions_save_path, exist_ok=True)

    for idx in range(target_size):
        pred_year = all_pred_inverse[:, idx]
        pred_df = pd.DataFrame(pred_year, columns=[f'Pred_Year{idx+1}'])
        pred_df.to_csv(os.path.join(predictions_save_path, f'pred_result_year{idx+1}.csv'), index=False)
        print(f"File saved {os.path.join(predictions_save_path, f'pred_result_year{idx+1}.csv')} successfully")

    # Save predictions to Excel
    original_file = "/content/Yellow River section interpolationⅡ - Deep Learning - 1/Yellow River09.xlsx"
    save_dir_excel = "/content/drive/MyDrive/Colab Notebooks/Thesis model/output data/CT/prediction_results_excel"
    os.makedirs(save_dir_excel, exist_ok=True)
    save_predictions_to_excel(all_pred_inverse, original_file, save_dir_excel, target_years=[22,23])

    print("All tasks completed!")

if __name__ == '__main__':
    main()
