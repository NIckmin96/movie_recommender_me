import argparse
import os
import pandas as pd
import numpy as np
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)
from torch.utils.data import DataLoader, random_split

from datasets import ContextModelDataset
from models import FM
from trainers import ContextModelTrainer

def main() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)
    parser.add_argument("--split_mode", default="random", type=str, help="random or user-by")
    
    # model args(FM)
    parser.add_argument("--model_name", default="FM", type=str)
    parser.add_argument("--emb_dim", default=20, type=int)
    parser.add_argument("--num_features", default=4, type=int)
    
    # train args
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--train_batch_size", default=20000, type=int)
    parser.add_argument("--valid_batch_size", default=10000, type=int)
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gp u_id")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    check_path(args.output_dir)
    
    
    fm_dataset = ContextModelDataset(args)
    model = FM(args)
    submission_dataset = ContextModelDataset(args, 'submission')
    
    # test_len = len(fm_dataset)//10
    valid_len = len(fm_dataset)//5
    train_len = len(fm_dataset)-(valid_len)
    fm_train, fm_valid = random_split(fm_dataset, [train_len, valid_len])
    fm_train_dl= DataLoader(fm_train, batch_size = args.train_batch_size, pin_memory = True)
    fm_valid_dl= DataLoader(fm_valid, batch_size = args.valid_batch_size, pin_memory = True)
    fm_test_dl = None
    # fm_test_dl= DataLoader(fm_test, batch_size = args.valid_batch_size, pin_memory = True)
    fm_sub_dl= DataLoader(submission_dataset, batch_size = args.valid_batch_size, pin_memory = True)
    
    fm_trainer = ContextModelTrainer(model, fm_train_dl, fm_valid_dl, fm_test_dl, fm_sub_dl, args)
    
    for epoch in range(1, args.epochs+1) : 
        fm_trainer.train(epoch)
        fm_trainer.valid(epoch)
        # if epoch%10 == 0 : 
        #     fm_trainer.valid(epoch)
    
    
    
    
    
if __name__ == '__main__' : 
    main()
        