#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday December 21 2020 at 18:42:24

@author: Tanmay Basu
"""

from data_extraction import data_extraction


#clf=data_extraction('/Users/basut/geometric_error_extraction/code/',model='keyword_matching')
#clf=data_extraction('/Users/basut/geometric_error_extraction/code/',model='tfidf',clf_opt='s',no_of_selected_terms=1000)
#clf=data_extraction('/Users/basut/geometric_error_extraction/code/',model='logentropy',clf_opt='s',no_of_selected_terms=1500)
#clf=data_extraction('/Users/basut/geometric_error_extraction/code/',model='doc2vec',clf_opt='s',no_of_selected_terms=1000)
clf=data_extraction('/Users/basut/geometric_error_extraction/code/',model='biobert')

clf.build_training_data()
clf.sentence_classification()
