#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description       :Preprocess the data and generate the pseudo data to train the module
@Date     :2022/08/31 10:47:00
@Author      :Lv Chuancheng
@version      :1.0
'''
from genericpath import exists
import random
import numpy as np
import json
import os
import re


class DataProcessor(object):
    """Process the training data.

    Attributes:
    """

    def __init__(self):
        pass

    def _read_jsonl(self, file_path):
        return [json.loads(line.strip()) for line in open(file_path, 'r').readlines()]
    
    def _write_jsonl(self, obj, file_path):
        f = open(file_path, 'w', encoding='utf-8')
        for line in obj:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    def _read_txt(self, file_path):
        return [i.strip() for i in open(file_path, 'r').readlines()]
        
    def _write_txt(self, obj, file_path):
        f = open(file_path, 'w', encoding='utf-8')
        for line in obj:
            f.write(line + '\n')

    def preprocess_data(self, data_path='../data/LOT', save_path='../data'):
        """Preprocess the LOT dataset.
        
        Args:
            data_path: path to the LOT dataset.
        """
        data = []
        splits = ['train', 'val', 'test']
        for split in splits:
            for line in self._read_jsonl(data_path + f'/outgen/{split}.jsonl'):
                data.append(line['story'])
            for line in self._read_jsonl(data_path + f'/clozet/{split}.jsonl'):
                pl = {i:line[f'plot{str(i)}'] for i in [0,1]}
                data.append(line['story'].replace('<mask>', pl[line['label']]))
            for line in self._read_jsonl(data_path + f'/plotcom/{split}.jsonl'):
                data.append(line['story'].replace('<mask>', line['plot']))
            for line in self._read_jsonl(data_path + f'/senpos/{split}.jsonl'):
                st = line['story'].split('<mask>')
                st = st[:line['label']+1] + [line['sentence']] + st[line['label']+1:]
        data = list(set(data))
        self._write_txt(data, save_path + '/data.txt')  
        
        
    def random_span_mask(self, text):
        """Span masks randomly for the origin text.
        """
        span_len = random.randint(round((len(text)-20)*0.1), round((len(text)-20)*0.3))
        start_idx = random.randint(20, len(text)-span_len)
        source = [text[:start_idx], text[start_idx+span_len:]]
        target = text[start_idx:start_idx+span_len]
        return span_len, source, target

    def process_text(self, source, target, mode):
        """Formulate the text into training instance of CPM-3
        """
        return {'source': source, 'target': target, 'mode': mode}
    
    def generate_pc_pseudo_data_random(self, source, target):
        data = self._read_jsonl(source)
        res = []
        for i in data:
            text = i['source'][0] + i['target']
            l, s, t = self.random_span_mask(text)
            res.append(self.process_text(s, t, 'lm'))
        self._write_jsonl(res, target)
        
    def generate_pc_pseudo_data_sent(self, source, target):
        data = self._read_txt(source)
        res = []
        for i in data:
            split = [j for j in i if j=='。']
            i = re.split('[。]', i)[:-1]
            assert(len(split) == len(i))
            idx = random.randint(0,len(i)-1)
            prefix = ''
            infix = ''
            postfix = ''
            for j in range(len(split)):
                if j == idx:
                    infix += i[j]+split[j]
                elif len(infix) > 0:
                    postfix += i[j]+split[j]
                else:
                    prefix += i[j]+split[j]
            res.append(self.process_text([prefix, postfix], infix, 'lm'))
        self._write_jsonl(res, target)


    def split_dataset(self, data_path):
        data = self._read_jsonl(data_path)
        random.shuffle(data)
        train_data = data[:int(len(data)*0.8)]
        val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
        test_data = data[int(len(data)*0.9):]
        self._write_jsonl(train_data, data_path.replace('.txt', '_train.txt'))
        self._write_jsonl(val_data, data_path.replace('.txt', '_val.txt'))
        self._write_jsonl(test_data, data_path.replace('.txt', '_test.txt'))

    
    def get_max_ppl_seq(self, nums, min_len = 3):
        if len(nums) < min_len:
            return [-1, -1]
        all_average = sum(nums)/len(nums)
        max_average = 0
        for start_idx in range(len(nums)-min_len):
            for end_idx in range(start_idx+min_len, len(nums)):
                average_num = sum(nums[start_idx:end_idx])/(end_idx-start_idx)
                if average_num > max_average:
                    max_average = average_num
                    res = [start_idx, end_idx]
        return max_average, res
    
    def detect_span_on_ppl(self, instance, min_len=3):
        ppls = instance['ppl']
        tokens = instance['tokens']
        for i in range(len(ppls)):
            print(ppls[i], tokens[i])
        max_average, res = self.get_max_ppl_seq(ppls, min_len)
        print(''.join(tokens[:res[0]]))
        print(''.join(tokens[res[0]:res[1]]))
        print(''.join(tokens[res[1]:]))
    
    def process_detector_data_beam(self, source_file, pred_file, target_file):
        source = self._read_jsonl(source_file)
        pred = self._read_txt(pred_file)
        assert(len(source) == len(pred))
        result = []
        for i in range(len(source)):
            prefix = source[i]['source'][0]
            postfix = source[i]['source'][1]
            infix = pred[i]
            target = source[i]['target']
            result.append(
                {
                    'prefix':prefix,
                    'infix':infix,
                    'postfix':postfix,
                    'target':target
                }
            )
        self._write_jsonl(result, target_file)
    
    def process_detector_data_contrastive(self, source_file, pred_file, target_file):
        source = self._read_jsonl(source_file)
        pred = self._read_txt(pred_file)
        assert(len(source) == len(pred))
        result = []
        for i in range(len(source)):
            prefix = source[i]['source'][0]
            postfix = source[i]['source'][1]
            infix = pred[i][:-len(postfix)]
            target = source[i]['target']
            result.append(
                {
                    'prefix':prefix,
                    'infix':infix,
                    'postfix':postfix,
                    'target':target
                }
            )
        self._write_jsonl(result, target_file)


if __name__ == '__main__':
    dp = DataProcessor()
    for i in ['train','val','test']:
        dp.process_detector_data_contrastive(
            f'/home/lvcc/CPM-3/data/pc_training_data_random/{i}.txt',
            f'/home/lvcc/CPM-3/data/pc_result_random/{i}_contrastive.txt',
            f'/home/lvcc/text-post-processing/data/random_contrastive_to_ground_truth/{i}.txt'
        )