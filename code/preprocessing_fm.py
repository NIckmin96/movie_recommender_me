import pandas as pd
import numpy as np
from functools import reduce
from scipy.sparse import csr_matrix
from tqdm import tqdm
import random
import os
import pickle

def preprocessing(merge_df): # value to index -> nn.Embedding
    merge_idx_df = pd.DataFrame()
    merge_idx_df['user_idx'] = pd.factorize(merge_df['user'])[0]
    merge_idx_df['item_idx'] = pd.factorize(merge_df['item'])[0]
    merge_idx_df['time_idx'] = pd.factorize(merge_df['time'])[0]
    merge_idx_df['year_idx'] = pd.factorize(merge_df['year'])[0]
    merge_idx_df['director_idx'] = pd.factorize(merge_df['director'])[0]
    merge_idx_df['genre_idx'] = pd.factorize(merge_df['genre'])[0]
    merge_idx_df['writer_idx'] = pd.factorize(merge_df['writer'])[0]
    
    user_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['user'])[0]), merge_df['user'].unique()))
    item_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['item'])[0]), merge_df['item'].unique()))
    time_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['time'])[0]), merge_df['time'].unique()))
    year_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['year'])[0]), merge_df['year'].unique()))
    director_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['director'])[0]), merge_df['director'].unique()))
    genre_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['genre'])[0]), merge_df['genre'].unique()))
    writer_idx_dict = dict(zip(np.unique(pd.factorize(merge_df['writer'])[0]), merge_df['writer'].unique()))
    
    index_dict_list = [user_idx_dict, item_idx_dict, time_idx_dict, year_idx_dict, director_idx_dict, genre_idx_dict, writer_idx_dict]
    
    return merge_idx_df, index_dict_list

def load_neg_sampling() : 
    
    import neg_sampling
    
    os.chdir(('/').join(str(__file__).split('/')[:-1]))
    if os.path.isfile('../data/neg_sample.pkl') : 
        pass
    else : 
        neg_sampling.main()
        
    with open("../data/neg_sample.pkl", 'rb') as f:
        neg_samples = pickle.load(f)
        
    return neg_samples