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
        span_len = random.randint(1, round(len(text)*0.3))
        start_idx = random.randint(1, len(text)-span_len)
        source = [text[:start_idx], text[start_idx+span_len:]]
        target = text[start_idx:start_idx+span_len]
        return source, target

    def process_text(self, source, target, mode):
        """Formulate the text into training instance of CPM-3
        """
        return {'source': source, 'target': target, 'mode': mode}
        
    
    def generate_lm_pseudo_data(self, data_path='../data/data.txt', save_path='../data/lm_pseudo_data.txt'):
        """Generate the lm pseudo data for training the CPM-3 model.
        """
        data = self._read_txt(data_path)
        pseudo_data = []
        for d in data:
            pseudo_data.append(self.process_text([d[:20],''], d[20:], 'lm'))
        self._write_jsonl(pseudo_data, save_path)
    
    def generate_pc_pseudp_data(self, data_path='/home/lvcc/CPM-3/data/data.txt', save_path='../CPM-3/data/pc_pseudo_data.txt'):
        """Generate the pc pseudo data for training the CPM-3 model.
        """
        data = self._read_txt(data_path)
        pseudo_data = []
        for d in data:
            source, target = self.random_span_mask(d)
            pseudo_data.append(self.process_text(source, target, 'lm'))
        self._write_jsonl(pseudo_data, save_path)
    
    def generate_prefix_pc_pseudo_data(self, source_data_path, data_path, save_path):
        source_data = self._read_jsonl(source_data_path)
        data = self._read_txt(data_path)
        pseudo_data = []
        assert(len(source_data) == len(data))
        for i in range(len(data)):
            source, target = self.random_span_mask(data[i])
            pseudo_data.append(self.process_text([source_data[i]['source'][0]+source[0], source[1]], target, 'lm'))
        self._write_jsonl(pseudo_data, save_path)


    def split_dataset(self, data_path):
        data = self._read_jsonl(data_path)
        random.shuffle(data)
        train_data = data[:int(len(data)*0.8)]
        val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
        test_data = data[int(len(data)*0.9):]
        self._write_jsonl(train_data, data_path.replace('.txt', '_train.txt'))
        self._write_jsonl(val_data, data_path.replace('.txt', '_val.txt'))
        self._write_jsonl(test_data, data_path.replace('.txt', '_test.txt'))

    def generate_seq_label_pseudo_data(self, source_data_path, output_data_path, save_path):
        """利用CPM-3的填空能力生成结果，然后结合原始数据，得到infix伪造的数据
        """
        
        
        source = [json.loads(i) for i in open(source_data_path, 'r').readlines()]
        output = [i.strip() for i in open(output_data_path).readlines()]
        prefix = [s['source'][0] for s in source]
        postfix = [s['source'][1] for s in source]
        infix = [output[i][:(len(output[i])-len(postfix[i]))] for i in range(len(output))]
        data = [{'prefix':prefix[i], 'postfix':postfix[i], 'infix':infix[i], 'truth':source[i]['target']} for i in range(len(prefix))]
        self._write_jsonl(data, save_path)
    
    def get_max_ppl_seq(self, nums, min_len = 3):
        if nums < min_len:
            return [-1, -1]
        max_average = 0
        for start_idx in range(len(nums)-min_len):
            for end_idx in range(i+min_len, len(nums)):
                average_num = sum(nums[start_idx:end_idx])
                if average_num > max_average:
                    max_average = average_num
                    res = [start_idx, end_idx]
        return max_average, res
        


if __name__ == '__main__':
    dp = DataProcessor()
    for s in ['train','val','test']:
        dp.generate_seq_label_pseudo_data(
            f'/home/lvcc/CPM-3/data/pc_data/pc_pseudo_data_0.3_{s}.txt',
            f'/home/lvcc/CPM-3/data/span_predict_train_data/{s}.txt',
            f'/home/lvcc/CPM-3/data/span_predict_train_data/{s}_pseudo.txt'
        )
