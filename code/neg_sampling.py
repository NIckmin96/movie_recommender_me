import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import random
import pickle
import os

def main() : 

    # load data
    df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
    user_idx = df.user.astype('category').cat.codes.values
    item_idx = df.item.astype('category').cat.codes.values
    df['interaction'] = 1
    sparse_mtx = csr_matrix((df.interaction, (df.user, df.item)))

    neg_samples = []
    for i,row in enumerate(sparse_mtx) : 
        # start = time.time()
        n_non_0 = len(np.nonzero(row)[0])
        zero_indices = np.argwhere(row==0)
        zero_indices[:,0] = i
        if n_non_0 > len(zero_indices) : 
            neg_indices = zero_indices.tolist()
        else : 
            neg_indices = random.sample(zero_indices.tolist(), n_non_0)
            
        neg_samples.extend(neg_indices)
        # end = time.time()
        # if (i!=0)&(i%10000 == 0) : 
            # print("time : ", end - start)
        
    with open('/opt/ml/bk/data/neg_sample.pkl', 'wb') as f:
        pickle.dump(neg_samples, f)
        
if __name__ == '__main__' : 
    main()