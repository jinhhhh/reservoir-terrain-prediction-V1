import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel(r'D:\PyCharm\paper_model\input_data\tezhengzhi-2.xlsx')

# 数据预处理
# 将年份列设置为索引，去掉第一行和“Unnamed: 0”列
data = data.drop(0)
data = data.set_index('Unnamed: 0')

# 转换为数值型数据
data = data.apply(pd.to_numeric)

# 选择特征列 (F1 到 F12)
features = data.columns

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建时间序列数据
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

# 使用前12年的数据预测未来的特征值
time_step = 1
X, y = create_dataset(scaled_data, time_step)

# 拆分数据集，80% 训练集，20% 测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM模型构建
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=12))  # 输出12个预测值（对应F1到F12）

# 编译和训练模型
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# 预测未来的数据
y_pred = model.predict(X_test)

# 将预测结果反标准化
y_pred_rescaled = scaler.inverse_transform(y_pred)

# 反标准化测试数据
y_test_rescaled = scaler.inverse_transform(y_test)

# 输出预测结果
print("预测值 (F1 到 F12):")
print(y_pred_rescaled)

# 计算预测误差
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f'均方误差: {mse}')

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:, 0], label="真实值 (F1)")
plt.plot(y_pred_rescaled[:, 0], label="预测值 (F1)", linestyle='--')
plt.legend()
plt.title('F1预测结果')
plt.show()

# 预测未来24-35年的数据
future_years = 12  # 预测未来12年（2024-2035）
future_predictions = []

# 使用最后的时间步数据作为初始输入
last_sequence = scaled_data[-time_step:]

for _ in range(future_years):
    # 预测下一年
    next_prediction = model.predict(last_sequence.reshape(1, time_step, -1))
    future_predictions.append(next_prediction[0])
    # 更新序列，加入新预测值，移除最早的一个值
    last_sequence = np.append(last_sequence[1:], next_prediction, axis=0)

# 反标准化未来预测结果
future_predictions_rescaled = scaler.inverse_transform(future_predictions)

# 创建未来年份的DataFrame
future_years_range = list(range(2024, 2036))  # 2024-2035年
future_df = pd.DataFrame(future_predictions_rescaled, columns=features)
future_df.insert(0, '年份', future_years_range)

# 保存预测结果到Excel文件
output_path = r'D:\PyCharm\paper_model\output_data\未来预测特征值结果_24-35年'  # 保存路径
future_df.to_excel(output_path, index=False)
print(f"未来24-35年的预测结果已保存到: {output_path}")