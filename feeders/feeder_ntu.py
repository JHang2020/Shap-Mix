import numpy as np

from torch.utils.data import Dataset

from feeders import tools
import random

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.avg_motion = self.data.mean(axis=0)[:,:64,...]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        #print(data_numpy.shape)
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Feeder_LT_ros(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, split_path=None, train_id_path=None, sample_name_path=None, class_balance=False):
        """
        long tailed data distribution with random over sampling
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        split_path: long tailed samples name list
        train_id_path: train_split ids coresponding to the sample_name_path
        sample_name_path: all sample names
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.split_path = split_path
        self.train_id_path = train_id_path
        self.sample_name_path = sample_name_path
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.class_balance = class_balance
        if class_balance:
            self.class_balance_data()
        else:
            self.balance_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            with open(self.split_path, 'r') as f:
                ske_list = [
                    line.strip() for line in f.readlines()
                ]
            with open(self.train_id_path, 'r') as f:
                train_id = [
                    int(line.strip()) for line in f.readlines()
                ]
            with open(self.sample_name_path, 'r') as f:
                sample_name_list = [
                    line.strip() for line in f.readlines()
                ]
            self.sample_name = [sample_name_list[train_id[i]] for i in range(len(self.data))]
            self.idx = []
            for i in range(len(self.data)):
                if self.sample_name[i] in ske_list:
                    self.idx.append(i)
            print('long tailed num: ', len(self.idx))
            self.data = self.data[self.idx]
            self.label = self.label[self.idx]

        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape

        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.avg_motion = self.data.mean(axis=0)[:,:64,...]
        self.idx_list = list(range(len(self.data)))

    def balance_data(self):
        if self.split != 'train':
            return
        
        max_num = 0
        for i in range(self.label.max()+1):
            num = (self.label==i).astype(np.int32).sum()
            max_num = max(max_num, num)

        for i in range(self.label.max()+1):
            idx = np.where(self.label==i)[0]
            num = len(idx)
            if num < max_num:
                sample_num = max_num - num
                new_s = random.choices(list(idx), k=sample_num)
                #print(new_s)
                self.idx_list += new_s
    
    def class_balance_data(self):
        if self.split != 'train':
            return
        self.class_idx = []
        self.class_num = self.label.max()+1
        for i in range(self.class_num):
            self.class_idx.append([])
            for idx in range(len(self.data)):
                if self.label[idx] == i:
                    self.class_idx[i].append(idx)
        self.idx_list = list(range(len(self.idx_list)*4))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.idx_list)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if self.class_balance:
            class_id = index % self.class_num
            index = random.sample(self.class_idx[class_id], 1)[0]
        else:
            index = self.idx_list[index]
        data_numpy = self.data[index]
        #print(data_numpy.shape)
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        #print(data_numpy.shape)
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class Feeder_LT_nosample(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, split_path=None, train_id_path=None, sample_name_path=None, class_balance=False):
        """
        long tailed data distribution
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        split_path: long tailed samples name list
        train_id_path: train_split ids coresponding to the sample_name_path
        sample_name_path: all sample names
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.split_path = split_path
        self.train_id_path = train_id_path
        self.sample_name_path = sample_name_path
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
  
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            with open(self.split_path, 'r') as f:
                ske_list = [
                    line.strip() for line in f.readlines()
                ]
            with open(self.train_id_path, 'r') as f:
                train_id = [
                    int(line.strip()) for line in f.readlines()
                ]
            with open(self.sample_name_path, 'r') as f:
                sample_name_list = [
                    line.strip() for line in f.readlines()
                ]
            self.sample_name = [sample_name_list[train_id[i]] for i in range(len(self.data))]
            self.idx = []
            for i in range(len(self.data)):
                if self.sample_name[i] in ske_list:
                    self.idx.append(i)
            print('long tailed num: ', len(self.idx))
            self.data = self.data[self.idx]
            self.label = self.label[self.idx]
            self.num_per_cls_dict = [0,]*(self.label.max()+1)
            for i in self.label:
                self.num_per_cls_dict[i] += 1

        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape

        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.avg_motion = self.data.mean(axis=0)[:,:64,...]
        self.idx_list = list(range(len(self.data)))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.idx_list)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        #print(data_numpy.shape)
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        #print(data_numpy.shape)
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class Feeder_LT_nosample_hybrid(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, split_path=None, train_id_path=None, sample_name_path=None, class_balance=False):
        """
        long tailed data distribution
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        split_path: long tailed samples name list
        train_id_path: train_split ids coresponding to the sample_name_path
        sample_name_path: all sample names
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.split_path = split_path
        self.train_id_path = train_id_path
        self.sample_name_path = sample_name_path
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.class_balance_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            with open(self.split_path, 'r') as f:
                ske_list = [
                    line.strip() for line in f.readlines()
                ]
            with open(self.train_id_path, 'r') as f:
                train_id = [
                    int(line.strip()) for line in f.readlines()
                ]
            with open(self.sample_name_path, 'r') as f:
                sample_name_list = [
                    line.strip() for line in f.readlines()
                ]
            self.sample_name = [sample_name_list[train_id[i]] for i in range(len(self.data))]
            self.idx = []
            for i in range(len(self.data)):
                if self.sample_name[i] in ske_list:
                    self.idx.append(i)
            print('long tailed num: ', len(self.idx))
            self.data = self.data[self.idx]
            self.label = self.label[self.idx]
            self.num_per_cls_dict = [0,]*(self.label.max()+1)
            for i in self.label:
                self.num_per_cls_dict[i] += 1

        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape

        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        self.avg_motion = self.data.mean(axis=0)[:,:64,...]
        self.idx_list = list(range(len(self.data)))

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def class_balance_data(self):
        if self.split != 'train':
            return
        self.class_idx = []
        self.class_num = self.label.max()+1
        for i in range(self.class_num):
            self.class_idx.append([])
            for idx in range(len(self.data)):
                if self.label[idx] == i:
                    self.class_idx[i].append(idx)
        self.idx_list = list(range(len(self.idx_list)))

    def __len__(self):
        return len(self.idx_list)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        class_id = index % self.class_num
        index2 = random.sample(self.class_idx[class_id], 1)[0]
        
        data_numpy = self.data[index]
        data_numpy2 = self.data[index2]
        #print(data_numpy.shape)
        label = self.label[index]
        label2 = self.label[index2]
        data_numpy = np.array(data_numpy)
        data_numpy2 = np.array(data_numpy2)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        valid_frame_num2 = np.sum(data_numpy2.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        data_numpy2 = tools.valid_crop_resize(data_numpy2, valid_frame_num2, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            data_numpy2 = tools.random_rot(data_numpy2)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        #print(data_numpy.shape)
        return data_numpy, label, data_numpy2, label2, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod