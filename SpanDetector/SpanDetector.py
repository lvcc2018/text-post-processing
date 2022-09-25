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

device = 'cuda:0'

class TagDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = [json.loads(i) for i in open(data_path).readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class RealTagDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = [i.strip() for i in open(data_path).readlines()]
    
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
    
    train_dataset = TagDataset('/home/lvcc/text-post-processing/data/random_contrastive_to_ground_truth/train.txt')
    valid_dataset = TagDataset('/home/lvcc/text-post-processing/data/random_contrastive_to_ground_truth/val.txt')
    test_dataset = TagDataset('/home/lvcc/text-post-processing/data/random_contrastive_to_ground_truth/test.txt')
    real_test_dataset = RealTagDataset('/home/lvcc/CPM-3/data/lm_result/test_final.txt')

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
    
    def real_collate_fn(batch):

        input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        input['labels'] = torch.zeros_like(input['input_ids'])
        for i in range(len(batch)):
            input['labels'][i][len(tokenizer(batch[i])['input_ids']):] = -100
        return input

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    real_test_dataloader = DataLoader(real_test_dataset, batch_size = 32, shuffle=False, collate_fn=real_collate_fn)
    total_steps = len(train_dataloader) * 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

    model.to(device)
    for epoch in range(0):
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
        model.eval()
        best_acc = 0
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
            if correct/total > best_acc:
                torch.save(model, '/home/lvcc/text-post-processing/best_eval.pt')
                best_acc = correct/total
    model = torch.load('/home/lvcc/text-post-processing/best_eval.pt')
    f = open('/home/lvcc/text-post-processing/result_real.txt','w')
    with torch.no_grad():
        correct = 0.
        total = 0.
        for idx, input in enumerate(real_test_dataloader):
            input = input.to(device)
            logits = model(**input).logits
            pred = logits.argmax(dim=-1)
            correct += (pred==input['labels']).sum().item()
            total += (input['labels']!=-100).sum().item()
            for i in range(input['labels'].shape[0]):
                res = ''
                flag = False
                flag2 = False
                for j in range(len(input['input_ids'][i])):
                    if input['labels'][i][j].item()!=-100:
                        if pred[i][j].item() == 0:
                            if flag == True:
                                flag = False
                                res += '[Pred_End]'
                        else:
                            if not flag:
                                flag = True
                                res += '[Pred_Begin]'
                        if input['labels'][i][j].item() == 0:
                            if flag2 == True:
                                flag2 = False
                                res += '[Truth_End]'
                        else:
                            if not flag2:
                                flag2 = True
                                res += '[Truth_Begin]'
                        res += tokenizer.decode(input['input_ids'][i][j])
                res += '\n'
                f.write(res)
        print('Test Acc: {}'.format(correct/total))

if __name__ == '__main__':
    finetune()
