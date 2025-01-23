from _utils import *
from config import *
import time
import sys
np.random.seed(28)

log_path = os.path.join(dataset_sparse_path, 'log.txt')
with open(log_path, "a") as f:
    f.write("\n-------------------------The current program is running in:")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write("-------------------------\n")
sys.stdout = Logger(log_path)

class GenerateSparse():

    def __init__(self):
        self.data_sparse_path = dataset_sparse_path
        self.dataset_path_list = [flow_sim_path, flow_zcity_path, pres_sim_path, pres_zcity_path]
        self.dataset_name_list = ['flow_sim', 'flow_zcity', 'pres_sim', 'pres_zcity']
        self.missing_name_list = ['random_0.3', 'random_0.6', 'long_range_0.3', 'block_0.3', 'mix_0.3', 'mix_0.5']
        return
    
    def generateSparseMatrix(self, missing_type_list, RM_missing_rate=None, LM_range_list=None, LM_missing_rate_list=None, BM_block_list=None, BM_missing_rate_list=None):
        tensor = self.dense_tensor.copy()
        for missing_type in missing_type_list:
            if missing_type == 'random':
                tensor = sparse_RM(tensor, RM_missing_rate)
            elif missing_type == 'long-range':
                tensor = sparse_LM(tensor, LM_range_list, LM_missing_rate_list)
            elif missing_type == 'block':
                tensor = sparse_BM(tensor, BM_block_list, BM_missing_rate_list)
        self.sparse_tensor = tensor
        self.sparse_mat = ten2mat(self.sparse_tensor, 0)
        self.sparse_mat_df = pd.DataFrame(self.sparse_mat.T, index=self.norm_data_df.index, columns=self.norm_data_df.columns)
        self.sparse_mat_df.to_csv(self.sparse_mat_for_completing_path, index=True, header=True)
        self.denorm_sparse_mat_df = reverse_one2hundred_normalization_without0(self.sparse_mat_df, self.raw_data_df)
        self.denorm_sparse_mat_df.to_csv(self.denorm_sparse_for_completing_path, index=True, header=True)
        return
    
    def main(self):
        for i in range(len(self.dataset_path_list)):
            print('\nStart %s dataset...' % self.dataset_name_list[i])
            dataset_name = self.dataset_name_list[i]
            dataset_name_sparse_path = os.path.join(self.data_sparse_path, dataset_name)
            create_folder(dataset_name_sparse_path)

            dataset_path = self.dataset_path_list[i]
            self.raw_data_df = read_csv_data(dataset_path)

            self.norm_data_df = one2hundred_normalization(self.raw_data_df)
            dense_mat = self.norm_data_df.values
            self.dense_tensor = dense_mat.T.reshape([dense_mat.shape[1], -1, int(24*60/15)]).transpose(0, 2, 1)
            self.data_norm_for_completing_path = os.path.join(dataset_name_sparse_path, 'data_norm_for_completing.csv')
            self.norm_data_df.to_csv(self.data_norm_for_completing_path, index=True, header=True)

            for j in range(len(self.missing_name_list)):
                missing_name = self.missing_name_list[j]
                dataset_missing_path = os.path.join(dataset_name_sparse_path, missing_name)
                create_folder(dataset_missing_path)

                self.sparse_mat_for_completing_path = os.path.join(dataset_missing_path, 'sparse_mat_for_completing.csv')
                self.denorm_sparse_for_completing_path = os.path.join(dataset_missing_path, 'denorm_sparse_for_completing.csv')

                if missing_name == 'random_0.3':
                    missing_type = 'random'
                    missing_rate = 0.3
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix([missing_type], missing_rate)
                if missing_name == 'random_0.6':
                    missing_type = 'random'
                    missing_rate = 0.6
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix([missing_type], missing_rate)
                if missing_name == 'long_range_0.3':
                    missing_type = 'long-range'
                    missing_rate = 0.3
                    LM_range_list, LM_missing_rate_list = [96, 48, 24], [0.10, 0.11, 0.12]
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix([missing_type], LM_range_list=LM_range_list, LM_missing_rate_list=LM_missing_rate_list)
                if missing_name == 'block_0.3':
                    missing_type = 'block'
                    missing_rate = 0.3
                    BM_block_list, BM_missing_rate_list = [2, 4, 6], [0.10, 0.11, 0.12]
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix([missing_type], BM_block_list=BM_block_list, BM_missing_rate_list=BM_missing_rate_list)
                if missing_name == 'mix_0.3':
                    missing_type = 'mix'
                    missing_rate = 0.3
                    RM_missing_rate = 0.1
                    LM_range_list, LM_missing_rate_list = [96, 48, 24], [0.03, 0.035, 0.04]
                    BM_block_list, BM_missing_rate_list = [2, 4, 6], [0.04, 0.045, 0.05]
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix(['random', 'long-range', 'block'], RM_missing_rate, LM_range_list, LM_missing_rate_list, BM_block_list, BM_missing_rate_list)
                if missing_name == 'mix_0.5':
                    missing_type = 'mix'
                    missing_rate = 0.5
                    RM_missing_rate = 0.2
                    LM_range_list, LM_missing_rate_list = [96, 48, 24], [0.06, 0.065, 0.07]
                    BM_block_list, BM_missing_rate_list = [2, 4, 6], [0.07, 0.075, 0.08]
                    print('\nStart %d %s missing...' % (missing_rate*100, missing_type))
                    self.generateSparseMatrix(['random', 'long-range', 'block'], RM_missing_rate, LM_range_list, LM_missing_rate_list, BM_block_list, BM_missing_rate_list)

        return

if __name__ == '__main__':
    generate_sparse = GenerateSparse()
    generate_sparse.main()
    sys.stdout.flush()