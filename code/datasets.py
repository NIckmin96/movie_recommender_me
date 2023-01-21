import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample


from functools import reduce
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
from tqdm import tqdm


class PretrainDataset(Dataset):
    def __init__(self, args, user_seq, long_sequence): # PretrainDatset(args, user_seq, long_sequence)
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length # default = 50
        self.part_sequence = [] # item "segment"
        self.split_sequence()

    def split_sequence(self): # user 별로 max sequence로 자르고 sub sequence로 나눠서 part_sequence에 append
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set / # 왜 이렇게 split? -> validation?
            for i in range(len(input_ids)): # for i in range(50)
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self): # sub sequence(segment)개수 return
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index]  # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p: # self.args.mask_p = 0.2(default)
                masked_item_sequence.append(self.args.mask_id) # mask_id = max_item(전체 itemset에서 가장 숫자가 큰 item)+1 -> 없는 item
                neg_items.append(neg_sample(item_set, self.args.item_size)) # item_size = max_item + 2
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id : start_id + sample_length]
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    
    
class ContextModelDataset(Dataset):
    # 1. random order
    # 2. user-by order
    def __init__(self, args, mode='train'):

        self.args = args
        self.mode = mode
        self.split_mode = args.split_mode
        self.dfs = self.load_data() # train_df,years,directors,genres,writers
        
        self.pos_df = self.dfs[0]
        self.pos_df['interaction'] = 1
        self.neg_df = self.load_neg_data()
        self.df = pd.concat([self.pos_df, self.neg_df], axis = 0)
        self.dfs[0] = self.df
        
        # if self.mode == 'train' : 
        #     self.merge_df = self.merge(self.dfs)
        #     self.merge_idx_df = self.encoding(self.merge_df)
            
        # elif self.mode == 'submission' : 
        #     self.dfs[0] = self.neg_df
        #     self.merge_df = self.merge(self.dfs)
        #     self.merge_idx_df = self.encoding(self.merge_df)
        
        self.merge_df = self.merge(self.dfs)
        self.merge_idx_df = self.encoding(self.merge_df)
        
        args.field_dims = np.max(self.merge_idx_df.loc[:,[col for col in self.merge_idx_df.columns if col not in ['interaction','time']]])+1
        # self.data = torch.tensor(self.merge_df[self.merge_df.columns.difference(['interaction'])].to_numpy(), dtype=torch.long)
        self.data = torch.tensor(self.merge_idx_df.loc[:,[col for col in self.merge_idx_df.columns if col not in ['interaction','time']]].to_numpy(), dtype=torch.long)
        # self.data = torch.tensor(self.merge_idx_df.loc[:,[col for col in self.merge_idx_df.columns if col not in ['interaction','time']]].groupby('user').to_numpy(), dtype=torch.long)
        # self.target = torch.tensor(self.merge_df['interaction'].to_numpy(), dtype=torch.float) # 0,1
        self.target = torch.tensor(self.merge_idx_df['interaction'].to_numpy(), dtype=torch.float) # 0,1
        
        if self.mode == 'submission' : 
            import pickle
            self.item_merge_df = self.merge(self.dfs[1:])
            with open('/opt/ml/bk/data/no_inter_items.pkl', 'rb') as f:
                self.no_inter_dict = pickle.load(f)
                
            self.no_inter_df = pd.DataFrame({'user':self.no_inter_dict.keys(), 'item':self.no_inter_dict.values()}).explode('item')
            
            
            
        
    def merge(self, dfs) :
        
        merge_df = reduce(lambda left, right:pd.merge(left, right, on='item', how='left'), dfs)
        
        return merge_df
    
    def encoding(self, merge_df) : 
        merge_idx_df = pd.DataFrame()
        merge_idx_df['user_idx'] = pd.factorize(merge_df['user'])[0]
        merge_idx_df['item_idx'] = pd.factorize(merge_df['item'])[0]
        # nan값이 있는 경우만 +1 (nan == -1)
        # merge_idx_df['time_idx'] = pd.factorize(merge_df['time'])[0] + 1
        merge_idx_df['year_idx'] = pd.factorize(merge_df['year'])[0] + 1
        merge_idx_df['director_idx'] = pd.factorize(merge_df['director'])[0] + 1
        merge_idx_df['genre_idx'] = pd.factorize(merge_df['genre'])[0] + 1
        merge_idx_df['writer_idx'] = pd.factorize(merge_df['writer'])[0] + 1
        merge_idx_df['interaction'] = merge_df['interaction']
        
        return merge_idx_df
    
    # FM
    def load_data(self):
        os.chdir(('/').join(str(__file__).split('/')[:-1]))
        df = pd.read_csv(os.path.join(self.args.data_dir, 'train_ratings.csv'))
        years = pd.read_csv(os.path.join(self.args.data_dir, 'years.tsv'), sep = '\t')
        directors = pd.read_csv(os.path.join(self.args.data_dir, 'directors.tsv'), sep = '\t')
        genres = pd.read_csv(os.path.join(self.args.data_dir, 'genres.tsv'), sep = '\t')
        # titles = pd.read_csv(os.path.join(self.args.data_dir, 'titles.tsv'), sep = '\t')
        writers = pd.read_csv(os.path.join(self.args.data_dir, 'writers.tsv'), sep = '\t')
        dfs = [df,years,directors,genres,writers]
        
        return dfs

    
    def load_neg_data(self) : 
        
        from preprocessing_fm import load_neg_sampling
    
        self.neg_samples = load_neg_sampling()
        neg_samples = np.array(self.neg_samples)
        neg_user = neg_samples[:,0]
        neg_item = neg_samples[:,1]
        neg_df = pd.DataFrame({'user':neg_user, 'item':neg_item, 'interaction':0})
        
        return neg_df
        
        
    def __len__(self):
        return len(self.target)
    
    # def __getitem__(self, index):
    #     if self.split_mode == 'random' :
    #         return self.data[index], self.target[index]
    
    def __getitem__(self, index): # get item by user group
        if self.split_mode == 'random' :
            return self.data[index], self.target[index]