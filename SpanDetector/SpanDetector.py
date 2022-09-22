#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description       :Detect the text span to be rewritten.
@Date     :2022/08/31 10:41:58
@Author      :Lv Chuancheng
@version      :1.0
'''

from re import T
import numpy as np
import datasets
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import (
    BertTokenizer, 
    BertConfig,
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

device = 'cuda:4'

class TagDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = [json.loads(i) for i in open(data_path).readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def finetune():
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_labels=2
    model = BertForTokenClassification.from_pretrained('bert-base-chinese', config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', config=config)

    model.to(device)
    
    train_dataset = TagDataset('/home/lvcc/text-post-processing/data/contrastive_search_to_ground_truth/train_pseudo.txt')
    valid_dataset = TagDataset('/home/lvcc/text-post-processing/data/contrastive_search_to_ground_truth/val_pseudo.txt')
    test_dataset = TagDataset('/home/lvcc/text-post-processing/data/contrastive_search_to_ground_truth/test_pseudo.txt')

    def collate_fn(batch):
        prefix = [i['prefix'] for i in batch]
        infix = [i['infix'] for i in batch]
        postfix = [i['postfix'] for i in batch]

        input = tokenizer([prefix[i]+infix[i]+postfix[i] for i in range(len(batch))], padding=True, truncation=True, return_tensors='pt')
        input['labels'] = torch.zeros_like(input['input_ids'])
        for i in range(len(batch)):
            prefix_length = len(tokenizer(prefix[i])['input_ids'])
            infix_length = len(tokenizer(infix[i])['input_ids'])
            postfix_length = len(tokenizer(postfix[i])['input_ids'])
            input['labels'][i][prefix_length-1:prefix_length+infix_length-3] = 1
            input['labels'][i][prefix_length+infix_length+postfix_length-4:] = -100
        return input

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    total_steps = len(train_dataloader) * 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

    model.to(device)
    for epoch in range(10):
        epoch_loss = 0.
        iter_loss = 0.
        
        model.train()
        with tqdm(total=len(train_dataloader)) as t:
            for idx, input in enumerate(train_dataloader):
                t.set_description('Training Epoch {}'.format(epoch))
                input = input.to(device)
                loss = model(**input).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                iter_loss = loss.item()
                t.set_postfix(loss=iter_loss)
                t.update(1)
                epoch_loss += iter_loss
        epoch_loss/=len(train_dataloader)
        print('Epoch {} loss: {}'.format(epoch, epoch_loss))
        model.save_pretrained('/home/lvcc/text-post-processing/model/epoch_{}.pt'.format(epoch))
        model.eval()
        with torch.no_grad():
            correct = 0.
            total = 0.
            for idx, input in enumerate(valid_dataloader):
                input = input.to(device)
                logits = model(**input).logits
                pred = logits.argmax(dim=-1)
                correct += (pred==input['labels']).sum().item()
                total += (input['labels']!=-100).sum().item()
            print('Valid Acc: {}'.format(correct/total))

if __name__ == '__main__':
    finetune()
