#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday December 21 2020 at 18:42:24

@author: Tanmay Basu
"""

from data_extraction import data_extraction

#clf=data_extraction('/home/xyz/data_extraction/',model='biobert')
#clf=data_extraction('/home/xyz/data_extraction/',model='keyword_matching')
#clf=data_extraction('/home/xyz/data_extraction/',model='tfidf',clf_opt='s',no_of_selected_terms=1000)
clf=data_extraction('/home/xyz/data_extraction/',model='entropy',clf_opt='s',no_of_selected_terms=1500)
#clf=data_extraction('/home/xyz/data_extraction/',model='doc2vec',clf_opt='s',no_of_selected_terms=1000)

clf.build_training_data()
clf.sentence_classification()
