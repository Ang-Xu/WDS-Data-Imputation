from _utils import *
import os

current_path = os.getcwd()

data_folder_path = os.path.join(current_path, 'data')
result_folder_path = os.path.join(current_path, 'result')

dataset_raw_path = os.path.join(data_folder_path, 'dataset_raw')
create_folder(dataset_raw_path)
flow_sim_raw_path = os.path.join(dataset_raw_path, 'flow_sim_raw.csv')
flow_zcity_raw_path = os.path.join(dataset_raw_path, 'flow_zcity_raw.csv')
pres_sim_raw_path = os.path.join(dataset_raw_path, 'pres_sim_raw.csv')
pres_zcity_raw_path = os.path.join(dataset_raw_path, 'pres_zcity_raw.csv')

dataset_path = os.path.join(data_folder_path, 'dataset')
create_folder(dataset_path)
flow_sim_path = os.path.join(dataset_path, 'flow_sim.csv')
flow_zcity_path = os.path.join(dataset_path, 'flow_zcity.csv')
pres_sim_path = os.path.join(dataset_path, 'pres_sim.csv')
pres_zcity_path = os.path.join(dataset_path, 'pres_zcity.csv')

dataset_sparse_path = os.path.join(data_folder_path, 'dataset_sparse')
create_folder(dataset_sparse_path)