#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description       :Preprocess the data and generate the pseudo data to train the module
@Date     :2022/08/31 10:47:00
@Author      :Lv Chuancheng
@version      :1.0
'''
import random

class DataProcessor(object):
    """Process the training data.
    
    Attributes:
    """
    def __init__(self):
        pass
    
    def random_span_mask(self, text:list, max_span_num:int, mask_ratio:float):
        """Span masks randomly for the origin text.
        
        Args:
            text: list of the tokens to be mask. 
        Returns:

        Raises:
        """
        text_length = len(text)
        mask_length = round(text_length * mask_ratio)
        span_number = random.randint(1, max(1, max_span_num))
        span_idx_pair = []
        for i in range(span_number):
            pass

        
        
    