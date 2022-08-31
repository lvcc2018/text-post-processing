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


class DataProcessor(object):
    """Process the training data.

    Attributes:
    """

    def __init__(self):
        pass

    def random_span_mask(self, tokens: list, max_span_num: int, mask_ratio: float = 0.2):
        """Span masks randomly for the origin text.

        Args:
            text: list of the tokens to be mask. 
            max_span_num: maximum of the span number.
            mask_ratio: ratio of the text to be masked.
        Returns:
            span_idx_pair: list of the start idx and end idxs.
        Raises:
        """
        text_length = len(tokens)
        mask_length = round(text_length * mask_ratio)
        span_number = random.randint(1, max(1, max_span_num))

        single_span_length = round(mask_length / span_number)
        span_idx_pair = []

        for i in range(span_number):
            random_span_length = min(max(1, np.random.normal(
                single_span_length, single_span_length * 0.1, 1)[0]), round(text_length/span_number))

            def get_start_idx():
                res = []
                for j in range(len(tokens)-random_span_length):
                    flag = True
                    for k in span_idx_pair:
                        if j in range(k[0], k[1]):
                            flag = False
                            break
                    if flag:
                        res.append(j)
                return res
            start_idx = random.choice(get_start_idx())
            span_idx_pair.append([start_idx, start_idx + random_span_length])
        return span_idx_pair

    def process_text(self, text):
        """Formulate the text into training instance of CPM-3
        
        Args:
        Returns:
        Raises:
        """
        
        
