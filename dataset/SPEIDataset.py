import numpy as np
import pandas as pd
import torch
import torch.utils.data


class SPEIDataset(torch.utils.data.Dataset):

    def __init__(self, seq_length=24, stride=1, month_type=1, infer=False):
        """
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        num_of_validation : number of validation dataset will be used
        infer : flag for test mode
        generate : flag for data generation mode
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        """
        # base files
        self.data_dir = f'/home/gongshuai/pythonProject/lstm4spei/data/SPEI-{month_type}.xlsx'
        # self.data_dir_1 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-1.xlsx'
        # self.data_dir_3 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-3.xlsx'
        # self.data_dir_6 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-6.xlsx'
        # self.data_dir_12 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-12.xlsx'
        # self.data_dir_24 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-24.xlsx'
        # self.data_dir_48 = '/home/gongshuai/pythonProject/lstm4spei/data/SPEI-48.xlsx'

        self.seq_length = seq_length
        self.stride = stride
        self.orig_seq_length = seq_length
        self.infer = infer

        self.data = self.load_xlsx(self.data_dir, skip_row=(int(month_type)-1))

        # Divide dataset in 7:3 for training and test
        count = int(self.data.shape[1] * 0.7)
        if not infer:
            self.data = self.data[:, :count]
        else:
            self.data = self.data[:, count:]

        self.data = self.preprocess(self.data)

        print(f'dataset_length = {len(self.data)}')


    def load_xlsx(self, data_file_path, skip_row=0):
        """
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        """
        # Load the data from the txt file
        print("Now processing: ", data_file_path)
        column_names = [
            'Year', 'Month', '51053', '51059', '51060', '51076', '51133', '51156', '51186', '51232', '51238', '51241',
            '51243', '51288', '51330', '51353', '51356', '51365', '51379', '51431', '51435', '51436', '51437', '51463',
            '51467', '51468', '51477', '51495', '51526', '51542', '51559', '51567', '51568', '51573', '51581', '51633',
            '51642', '51656', '51701', '51704', '51705', '51708', '51711', '51765', '51777', '51804', '51811', '51815',
            '51818', '51826', '51828', '51839', '51855', '51931', '52101', '52118', '52203']
        file_data = pd.read_excel(data_file_path, dtype=float)
        data = file_data.iloc[skip_row:, 2:len(column_names)]
        data = np.array(data)
        data = torch.from_numpy(data).permute(1, 0)  # (站点， SPEI)
        return data

    def preprocess(self, data):
        result = []
        b, l = data.shape
        count = (l - self.seq_length) // self.stride + 1
        for i in range(b):
            for j in range(count):
                result.append(data[i][j: j+self.seq_length])
            if ((count - 1) * self.stride + self.seq_length) < l:
                result.append(data[i][l-self.seq_length:l])
        result = torch.stack(result, dim=0)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
