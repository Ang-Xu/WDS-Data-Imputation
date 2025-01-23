import os
import sys
import yaml
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(28)

def read_csv_data(data_path):
    data_df = pd.read_csv(data_path, index_col=0, header=0)
    data_df.index = pd.to_datetime(data_df.index)
    return data_df

def ten2mat(tensor, mode):  # tensor to matrix
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):  # matrix to tensor
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order = 'F'), 0, mode)

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0] * 100

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def compute_smape(var, var_hat):
    return 1/len(var) * np.sum(2 * np.abs(var_hat - var) / (np.abs(var) + np.abs(var_hat)) * 100)

# The 0 element is not included in the normalization because the 0 element represents a missing value, not a true 0. Normalize the other elements to the interval [1, 100].
def one2hundred_normalization(data_df):
    # Convert data_df to float64 type, since data_df may contain int type data
    data_df = data_df.astype('float64')
    data_df = data_df.copy()
    data_df[data_df != 0.0] = preprocessing.minmax_scale(data_df[data_df != 0], feature_range=(1, 100))
    return data_df

# Backnormalization, no backnormalization for 0 elements
def reverse_one2hundred_normalization_without0(data_df, raw_data_df):
    data_df = data_df.copy()
    data_df[data_df != 0] = preprocessing.MinMaxScaler(feature_range=(1, 100)).fit(raw_data_df[raw_data_df != 0]).inverse_transform(data_df[data_df != 0])
    return data_df

# Backnormalization
def reverse_one2hundred_normalization(data_df, raw_data_df):
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 100))
    scaler.fit(raw_data_df[raw_data_df != 0])
    denorm_data = scaler.inverse_transform(data_df)
    denorm_data_df = pd.DataFrame(denorm_data, index=data_df.index, columns=data_df.columns)
    return denorm_data_df

def create_folder(folder_path):
    if isinstance(folder_path, str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    elif isinstance(folder_path, list):
        for path in folder_path:
            if not os.path.exists(path):
                os.makedirs(path)

def sparse_RM(raw_tensor, missing_rate):
    
    dim1, dim2, dim3 = raw_tensor.shape
    sparse_tensor = raw_tensor * np.round(np.random.rand(dim1, dim2, dim3) + 0.5 - missing_rate)  # 随机缺失
    return sparse_tensor

def sparse_LM(raw_tensor, range_list, missing_rate_list):
    raw_mat = ten2mat(raw_tensor, 0)
    sparse_mat = raw_mat.copy()

    for range, missing_rate in zip(range_list, missing_rate_list):
        dense_tensor = sparse_mat.reshape([raw_mat.shape[0], -1, range]).transpose(0, 2, 1)
        dim1, dim2, dim3 = dense_tensor.shape
        sparse_tensor = dense_tensor * np.round(np.random.rand(dim1, dim3) + 0.5 - missing_rate)[:, None, :]
        sparse_mat = ten2mat(sparse_tensor, 0)

    sparse_tensor_return = sparse_mat.reshape([raw_mat.shape[0], -1, int(24*60/15)]).transpose(0, 2, 1)
    return sparse_tensor_return

def sparse_BM(raw_tensor, block_list, missing_rate_list):

    dim1, dim2, dim3 = raw_tensor.shape
    dim_time = dim2 * dim3
    sparse_tensor = raw_tensor.copy()

    for block, missing_rate in zip(block_list, missing_rate_list):
        vec = np.random.rand(int(dim_time / block))
        temp = np.array([vec] * block)
        vec = temp.reshape([dim2 * dim3], order = 'F')
        sparse_tensor = mat2ten(ten2mat(sparse_tensor, 0) * np.round(vec + 0.5 - missing_rate)[None, :], np.array([dim1, dim2, dim3]), 0)

    return sparse_tensor

def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return yaml_data

def write_yaml(yaml_path, yaml_data):
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True)

def plt_result(raw_data_df, data_complete_df, sparse_mat_df, plt_result_path, mape_list, rmse_list, smape_list):
    columns = data_complete_df.columns
    data_len = 96 * 7
    plt.ioff()
    fig, axes = plt.subplots(len(columns), 1, figsize=(15, 4 * len(columns)))
    for i in range(len(columns)):
        column_name = columns[i]
        raw_data_value = raw_data_df.iloc[-data_len:, i]
        data_complete_value = data_complete_df.iloc[-data_len:, i]
        sparse_for_completing_value = sparse_mat_df.iloc[-data_len:, i]

        condition = (sparse_for_completing_value == 0) & (raw_data_value != 0)
        mask = np.zeros_like(condition, dtype=bool)
        mask[np.where(condition)] = True
        data_complete_value = data_complete_value.where(mask, np.nan)

        sns.lineplot(data=raw_data_value, ax=axes[i], label='raw data', color='red')
        sns.scatterplot(data=data_complete_value, ax=axes[i], marker='o', label='sparse complete', color='blue')
        axes[i].set_title(column_name + '  mape: %.3f, rmse: %.3f, smape: %.3f' % (mape_list[i], rmse_list[i], smape_list[i]))

    fig.tight_layout()
    fig.savefig(plt_result_path, dpi=100, bbox_inches='tight')
    plt.close()
    print('plt_result_path: ', plt_result_path)
    return

def result_analysis(dense_df, sparse_df, mat_df):
    assert dense_df.shape == sparse_df.shape == mat_df.shape
    columns = dense_df.columns
    mape_list = []
    rmse_list = []
    smape_list = []
    training_num_list = []
    dense_test_list = []
    hat_test_list = []
    imput_num_list = []
    for i in range(len(columns)):
        column_name = columns[i]
        dense_value = dense_df.loc[:, column_name].values
        sparse_value = sparse_df.loc[:, column_name].values
        hat_value = mat_df.loc[:, column_name].values
        pos_test = np.where((dense_value != 0) & (sparse_value == 0))
        pos_imput = np.where((dense_value == 0))
        dense_test = dense_value[pos_test]
        hat_test = hat_value[pos_test]
        mape = compute_mape(dense_test, hat_test)
        rmse = compute_rmse(dense_test, hat_test)
        smape = compute_smape(dense_test, hat_test)
        mape_list.append(mape)
        rmse_list.append(rmse)
        smape_list.append(smape)
        training_num_list.append(len(dense_test))
        imput_num_list.append(len(pos_imput[0]))
        dense_test_list.append(dense_test.tolist())
        hat_test_list.append(hat_test.tolist())
    
    mape_list = np.array(mape_list).round(3)
    rmse_list = np.array(rmse_list).round(3)
    smape_list = np.array(smape_list).round(3)
    training_num_list = np.array(training_num_list)
    imput_num_list = np.array(imput_num_list)
    dense_test_list = np.array([j for i in dense_test_list for j in i])
    hat_test_list = np.array([j for i in hat_test_list for j in i])
    global_mape, global_rmse, global_smape = compute_mape(dense_test_list, hat_test_list), compute_rmse(dense_test_list, hat_test_list), compute_smape(dense_test_list, hat_test_list)
    print('\nafter denormalization, total num: %d, global mape: %.4f, global rmse: %.4f, global smape: %.4f\n' % (len(dense_test_list), global_mape, global_rmse, global_smape))
    result_df = pd.DataFrame(np.array([mape_list, rmse_list, smape_list]).T, index=columns, columns=['mape', 'rmse', 'smape'])
    result_df['trainNum'] = training_num_list
    result_df['tPer'] = (training_num_list / len(dense_df)).round(3)
    result_df['missingNum'] = imput_num_list
    result_df['mPer'] = (imput_num_list / len(dense_df)).round(3)
    print(result_df)
    return mape_list, rmse_list, smape_list

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.log.flush()