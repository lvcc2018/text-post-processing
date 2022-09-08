#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description       :Preprocess the data and generate the pseudo data to train the module
@Date     :2022/08/31 10:47:00
@Author      :Lv Chuancheng
@version      :1.0
'''
import random
import numpy as np
import json


class DataProcessor(object):
    """Process the training data.

    Attributes:
    """

    def __init__(self):
        pass

    def _read_jsonl(self, file_path):
        return [json.loads(line.strip) for line in open(file_path, 'r').readlines()]
    
    def _write_jsonl(self, obj, file_path):
        f = open(file_path, 'w', encoding='utf-8')
        for line in obj:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    def preprocess_data(self, data_path='../data/LOT', save_path='../data'):
        """Preprocess the LOT dataset.
        
        Args:
            data_path: path to the LOT dataset.
        """
        data = []
        splits = ['train', 'valid', 'test']
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
                st = st[:line['label']+1] + line['sentence'] + st[line['label']+1:]
        data = list(set(data))
        self._write_jsonl(data, save_path + '/data.jsonl')          
        
        

    def random_span_mask(self, tokens: list, max_span_num: int, mask_ratio: float = 0.2):
        """Span masks randomly for the origin text.

        Args:
            
        Returns:
            
        Raises:
        """
        pass

    def process_text(self, text):
        """Formulate the text into training instance of CPM-3

        Args:
        Returns:
        Raises:
        """
        pass


if __name__ == '__main__':
    dp = DataProcessor()
    dp.preprocess_data()
