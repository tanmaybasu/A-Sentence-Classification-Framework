#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday December 21 2020 at 18:42:24

@author: Tanmay Basu
"""

from data_extraction import data_extraction


#clf=sentence_classification('/home/xyz/sentence_classification/',model='keyword_matching',threshold=0.5)
#clf=sentence_classification('/home/xyz/sentence_classification/',model='tfidf',clf_opt='s',no_of_selected_terms=1000,threshold=0.5)
clf=sentence_classification('/home/xyz/sentence_classification/',model='entropy',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)
#clf=sentence_classification('/home/xyz/sentence_classification/',model='doc2vec',vec_len=20,clf_opt='s',no_of_selected_terms=1000,threshold=0.5)
#clf=sentence_classification('/home/xyz/sentence_classification/',model='bert',model_source='monologg/biobert_v1.1_pubmed',threshold=0.5)

#clf.build_training_data()
clf.sentence_classification()
