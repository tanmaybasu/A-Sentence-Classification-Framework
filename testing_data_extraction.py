#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday December 21 2020 at 18:42:24

@author: Tanmay Basu
"""

from data_extraction import data_extraction

#clf=data_extraction('/home/xyz/data_extraction/',model='biobert',model_source='monologg/biobert_v1.1_pubmed',threshold=0.5)
#clf=data_extraction('/home/xyz/data_extraction/',model='keyword_matching',threshold=0.5)
#clf=data_extraction('/home/xyz/data_extraction/',model='tfidf',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)
clf=data_extraction('/home/xyz/data_extraction/',model='entropy',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)
#clf=data_extraction('/home/xyz/data_extraction/',model='doc2vec',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)

clf.build_training_data()
clf.sentence_classification()
