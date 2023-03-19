import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
import numpy as np


class HumeDataset(Dataset):
    def __init__(self, data, partition, feature, max_len=30):
        super(HumeDataset, self).__init__()
        self.partition = partition

        if len(data) > 1 and partition=='train':
            if feature[0] == 'DeepSpectrum':
                # 将音频特征中存在但是视觉特征不存在的数据去掉，即得到公共部分
                delete_id = [
                    590, 592, 4805, 5906, 6613, 6763, 8756, 9297, 11797, 13751,
                    14028, 14044, 14832
                ]
                print(f'Delete id {delete_id}')
                for id in delete_id:
                    data[0][partition]['feature'].pop(id)
                    data[0][partition]['label'].pop(id)
                    data[0][partition]['meta'].pop(id)
                assert len(data[0][partition]['feature']) == len(
                    data[1][partition]['feature'])
            else:
                print("使用了多个特征， 第一个特征不是音频")
                
        features = []
        for i in range(len(data)):
            features.append(data[i][partition]['feature'])

        labels = data[0][partition]['label']
        metas = data[0][partition]['meta']
        self.feature_dim = [feature[0].shape[-1] for feature in features]
        self.n_samples = len(features[0])
        feature_lens = []
        label_lens = []
        for feature in features:
            feature_len = []
            for feature_item in feature:
                feature_len.append(len(feature_item))
            feature_lens.append(feature_len)
        for label in labels:
            if label.ndim == 1:
                label_lens.append(1)
            else:
                label_lens.append(label.shape[0])

        # 指定feature length
        # max_feature_len = [np.max(np.array(feature_len)) for feature_len in feature_lens]
        max_feature_len = [max_len] * len(data)
        self.feature_len_in_one_step = np.max(np.array(max_feature_len))
        # if self.feature_len_in_one_step:
        #     index = np.argmax(feature_lens[0])
        #     print(self.feature_len_in_one_step)
        #     print(feature_lens[0][index])
        #     print(metas[index])

        max_label_len = np.max(np.array(label_lens))
        if max_label_len > 1:
            assert(max_feature_len==max_label_len)

        self.feature_lens = [torch.tensor(feature_len) for feature_len in feature_lens]

        # 截取或填充数据
        for i in range(len(features)):
            for j in range(len(features[i])):
                if features[i][j].shape[0] > max_feature_len[i]:
                    features[i][j] = features[i][j][:max_feature_len[i]]
                else:
                    features[i][j] = np.pad(features[i][j], pad_width=((0, max_feature_len[i]-features[i][j].shape[0]),(0,0)))

        # features = [[np.pad(f, pad_width=((0, max_feature_len[i]-f.shape[0]),(0,0))) for f in features[i]] for i in range(len(features))]

        self.features = [[torch.tensor(f.astype(np.float), dtype=torch.float) for f in feature] for feature in features]
        # if n-to-n task like stress
        if max_label_len > 1:
            labels = [np.pad(l, pad_width=((0, max_label_len-l.shape[0]),(0,0))) for l in labels]
        self.labels = [torch.tensor(l, dtype=torch.float) for l in labels]
        self.metas = [np.pad(meta, pad_width=((0,max_label_len-meta.shape[0]),(0,0)), mode='empty') for meta in metas]

        self.metas = [m.astype(np.object).tolist() for m in self.metas]

        self.max_feature_len = max_feature_len


    def get_max_feature_len(self):
        return self.max_feature_len

    def get_feature_dim(self):
        return self.feature_dim

    def get_feature_len_in_one_step(self):
        return self.feature_len_in_one_step

    def get_labels(self):
        return self.labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        '''
        :param idx:
        :return: feature, feature_len, label, meta with
            feature: tensor of shape seq_len, feature_dim
            feature_len: int tensor, length of the feature tensor before padding
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len,1))
            meta: list of lists containing corresponding meta data
        '''
        feature = [feature[idx] for feature in self.features]
        feature_len = [feature_len[idx] for feature_len in self.feature_lens]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = feature, feature_len[0], label, meta
        return sample


def custom_collate_fn(data):
    '''
    Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
    :param data:
    :return:
    '''
    tensors = [d[:3] for d in data]
    np_arrs = [d[3] for d in data]
    coll_features, coll_feature_lens, coll_labels = default_collate(tensors)
    np_arrs_coll = np.row_stack(np_arrs)
    return coll_features, coll_feature_lens, coll_labels, np_arrs_coll
