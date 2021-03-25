#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday December 19 2020 at 16:16:19

@author: Tanmay Basu
"""

import csv,os,re,sys
import fitz
import nltk
import numpy as np
import torch
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import LogEntropyModel
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec,TaggedDocument 
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


en_stopwords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
nltk_stopwords=list(set(stopwords.words('english')))
for word in nltk_stopwords:
    if word not in en_stopwords:
        en_stopwords.append(word)

# Class for Torch Model
class get_torch_data_format(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# Main Class
         
class data_extraction():
     def __init__(self,path='/home/xyz/data_extrcation/',model='entropy',model_source=None,clf_opt='s',no_of_selected_terms=None,threshold=0.5):
        self.path = path
        self.model = model
        self.model_source=model_source
        self.clf_opt=clf_opt
        self.no_of_selected_terms=no_of_selected_terms
        if self.no_of_selected_terms!=None:
            self.no_of_selected_terms=int(self.no_of_selected_terms) 
        self.threshold=float(threshold)
# PDF to text conversion
     def pdf_to_text(self,file):        
        try:
            doc = fitz.open(file, filetype = "pdf")        # PDF to text conversion  
            texts=[] 
            for page in doc:
                texts.append(page.getText()) 
            text=''.join(texts)
        except:
            text='' 
        return text

# Text refinement
     def text_refinement(self,text='hello'):
    #    text = re.sub(r'[^!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\w]+',' ', text)     # Remove special characters e.g., emoticons-😄. The ^ in the beginning ensures that all the natural characters will be kept. 
    #    text = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', text)                          # Remove special characters
        text=re.sub(r'[?!]', '\.', text)                                           # Replace '?' with '.' to properly identify floating point numbers 
        text=re.sub(r'([a-zA-Z0-9])([\),.!?;-]+)([a-zA-Z])', r'\1\2 \3', text )    # Space between delimmiter and letter   
        text=re.sub(r'([a-z])([\.])([\s]*)([a-z])', r'\1 \3\4', text)              # Reomove '.' between two lowercase letters e.g., et al. xxx
        text=re.sub(r'([0-9]+)([\.]+)([0-9]+)([\.]+)([0-9]+)', r'\1-\3-\5', text)  # Reomove '.' between three decimal numbers e.g., et 000.55.66
        text=re.sub(r'([a-z])([\.]*)([0-9])', r'\1\2 \3', text)                    # Space between letter and no.    
        text=re.sub(r'(\s)([a-z0-9]+)([A-Z])([\w]+)', r'\1\2. \3\4', text)         # Put a '.' after a lowercase letter/number followed by Uppercase e.g., drains removed by day threeHe continued to 
        text=re.sub(r'([a-z0-9])([\n]+)([A-Z])', r'\1\. \3', text)                 # Put a between lowercase letter/number, \n and uppercase letter e.g., xxx5 \n Yyy
        text=re.sub(r'(\.)([\s]*)([\.]+)', r'\1', text)                            # Removing extra '.'s, if any 
        return text

# Final score calculation
     def get_sent_score(self,sent,phrase):
        sent_tokens = nltk.word_tokenize(sent.lower().strip('.'))
        sent_tokens = [item.rstrip('s') for item in sent_tokens]            # Converting the plurals to singulars  
        target_tokens = nltk.word_tokenize(phrase.lower().strip('.'))              
        target_tokens = [item.rstrip('s') for item in target_tokens]        # Converting the plurals to singulars             
        score = 0   
        # sent_sim is the similarity measure
        for token in target_tokens:
            if token in sent_tokens and token not in stopwords.words('english'):    # Discarding the stopwords 
                score += 1
        if score!=0:
            score = float(score)/len(target_tokens)      
        return score
    
# Check if a sentence is relevant to the data element 
     def check_relevant_sentences(self,sentence,keywords):
        phrase_score=[]; total_score=0.0
        sentence = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', sentence)    # Remove special character 
        for phrase in keywords:
            score=0.0;
            score=self.get_sent_score(sentence,phrase)
            total_score+=score
            if score>=self.threshold:
                tmp=[]
                tmp.append(phrase)
                tmp.append(score)
                phrase_score.append(tmp)
        if phrase_score!=[]:
            phrase_score.sort(key=lambda x: x[1], reverse=True)     # Sorting the phrases according to ascending order of similarity scores 
            return phrase_score
        if total_score==0:
            tmp=[]; 
            tmp.append(phrase)
            tmp.append(score)
            phrase_score.append(tmp)
        return phrase_score
   
# Building training corpus  
     def build_training_data(self):
         if os.path.isfile(self.path+'keywords.txt') and os.path.getsize(self.path+'keywords.txt') > 0:                      
             fk=open(self.path+'keywords.txt', "r") 
             keywords = list(csv.reader(fk,delimiter='\n'))
             keywords = [item for sublist in keywords for item in sublist]
             fk.close()
         else:                                                  # If the keywords file does not exist or is empty then exit 
             print('Either keywords.txt does not exist or it is empty \n')
             sys.exit(0)
    # If the training data directory does not exist then exit
         if not os.path.isdir(self.path+'training_data'):
            print('The directory training_data does not exist \n')
            sys.exit(0)
         trn_files=os.listdir(self.path+'training_data')
         if trn_files==[]:
             print('There is no training samples in the directory \n')
         else:
             count=0;
             print('############### Preparing Training Data ############### \n')
             fp=open(self.path+'training_relevant_class_data.csv',"w")
             fn=open(self.path+'training_irrelevant_class_data.csv',"w")  
             for item in trn_files:
                if item.find('.pdf')>0:                       # Checking if it is a PDF file
                    count+=1
                    text=self.pdf_to_text(self.path+'training_data/'+item) 
                    text=self.text_refinement(text)                     # Cleaning text file 
                    fp.write(item.rstrip('.pdf')+',')
                    fn.write(item.rstrip('.pdf')+',')
                    if text!='':
                        text=re.sub(r',', r'', text)      # replacing , by ; to build the csv properly 
                        text=re.sub(r'\n', r'\\n ', text)    # replacing \n by '\\n ' to build the csv properly  
                        sentences=nltk.sent_tokenize(text)
                        for sentence in sentences:
                            if sentence!='':
                                phrase_score=self.check_relevant_sentences(sentence,keywords)
                                if phrase_score!=[]:
                                    ln=len(nltk.word_tokenize(phrase_score[0][0]))
             # Check if there is a floating point number 
                                    if ln>2 and phrase_score[0][1]>=self.threshold and re.findall(r'\d+\.\d+', sentence)!=[]:    
                                        fp.write(sentence+' ')
                                    elif ln<=2 and phrase_score[0][1]>self.threshold and re.findall(r'\d+\.\d+', sentence)!=[]: 
                                        fp.write(sentence+' ')
            # Check if there is no number and given keywords 
                                    elif phrase_score[0][1]==0 and re.findall(r'\d+\.\d+', sentence)==[]: 
                                        fn.write(sentence+' ') 
                    else:
                        print('Empty text for file: '+item+'\n') 
                    fp.write('\n')
                    fn.write('\n')
             fp.close()
             fn.close()

    # Selection of classifiers  
     def classification_pipeline(self):        
        # Logistic Regression 
        if self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            ext2='logistic_regression'
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
        # Linear SVC 
        elif self.clf_opt=='ls':   
            print('\n\t### Training Linear SVC Classifier ### \n')
            ext2='linear_svc'
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,1,2,10,50,100),
            }         
        # Multinomial Naive Bayes
        elif self.clf_opt=='n':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            ext2='naive_bayes'
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }            
        # Random Forest 
        elif self.clf_opt=='r':
            print('\n\t ### Training Random Forest Classifier ### \n')
            ext2='random_forest'
            clf = RandomForestClassifier(criterion='gini',max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__n_estimators':(30,50,100,200),
            'clf__max_depth':(10,20),
            }          
        # Support Vector Machine  
        elif self.clf_opt=='s': 
            print('\n\t### Training Linear SVM Classifier ### \n')
            ext2='svm'
            clf = svm.SVC(kernel='linear', class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,0.5,1,2,10,50,100),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters,ext2        
    
# TFIDF model    
     def tfidf_training_model(self,trn_data,trn_cat):
        print('\n ***** Building TF-IDF Based Training Model ***** \n')
        clf,clf_parameters,ext2=self.classification_pipeline() 
        print('No terms \t'+str(self.no_of_selected_terms))
        if self.no_of_selected_terms==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([
                ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
                ('clf', clf),]) 
        else:
            try:                                        # To use selected terms of the vocabulary
                pipeline = Pipeline([
                    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
                    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_terms)),                         # k=1000 is recommended 
                    ('clf', clf),]) 
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
    # Fix the values of the parameters using Grid Search and cross validation on the training samples 
        feature_parameters = {
        'vect__min_df': (2,3),
        'vect__ngram_range': ((1, 2),(1,3)),  # Unigrams, Bigrams or Trigrams
        }
        parameters={**feature_parameters,**clf_parameters}
    
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_  
        print(clf)            
        return clf,ext2

# Doc2Vec model    
     def doc2vec_training_model(self,ln,trn_data,trn_cat):
        print('\n ***** Building Doc2Vec Based Training Model ***** \n')
        tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trn_data)]
        max_epochs = 10       
        trn_model = Doc2Vec(vector_size=int(ln),alpha=0.025,min_alpha=0.00025,min_count=1,dm =1)
        trn_model.build_vocab(tagged_data)  
        print('Number of Training Samples {0}'.format(trn_model.corpus_count))   
        for epoch in range(max_epochs):
           print('Doc2Vec Iteration {0}'.format(epoch))
           trn_model.train(tagged_data,
                       total_examples=trn_model.corpus_count,
                       epochs=100) 
           # decrease the learning rate
           trn_model.alpha -= 0.0002
        trn_vec=[]
        for i in range(0,len(trn_data)):
              vec=[] 
              for v in trn_model.docvecs[i]:
                  vec.append(v)
              trn_vec.append(vec)
    # Classificiation and feature selection pipelines
        clf,clf_parameters,ext2=self.classification_pipeline() 
        pipeline = Pipeline([('clf', clf),])       
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
        print(clf)  
        return clf,ext2,trn_model
               
# Keyword_matching model    
     def keyword_matching_model(self,tst_data):
        predicted=[1 for i in range(0,len(tst_data))]
        if os.path.isfile(self.path+'keywords.txt') and os.path.getsize(self.path+'keywords.txt') > 0:    
            fk=open(self.path+'keywords.txt', "r") 
            keywords = list(csv.reader(fk,delimiter='\n'))
            keywords = [item for sublist in keywords for item in sublist]
            fk.close()
        else:                                                  # If the keywords file does not exist or is empty then exit 
            print('Either keywords.txt does not exist or it is empty \n')
            sys.exit(0)
        num=0;
        for sentence in tst_data:
            if sentence!='':
                phrase_score=self.check_relevant_sentences(sentence,keywords)
                if phrase_score!=[]:
                    ln=len(nltk.word_tokenize(phrase_score[0][0]))
# Check if there is a floating point number 
                    if ln>2 and phrase_score[0][1]>=self.threshold and re.findall(r'\d+\.\d+', sentence)!=[]:    
                        predicted[num]=0
                    elif ln<=2 and phrase_score[0][1]>self.threshold and re.findall(r'\d+\.\d+', sentence)!=[]: 
                        predicted[num]=0
            num+=1   
        return predicted
     
# Entropy model    
     def entropy_training_model(self,trn_data,trn_cat): 
        print('\n ***** Building Entropy Based Training Model ***** \n')
        trn_vec=[]; trn_sentences=[]; 
        for sentence in trn_data:
            sentence=nltk.word_tokenize(sentence.lower())
            trn_sentences.append(sentence)                       # Training sentences broken into words
        trn_dct = Dictionary(trn_sentences)
        corpus = [trn_dct.doc2bow(row) for row in trn_sentences]
        trn_model = LogEntropyModel(corpus)
        no_of_terms=len(trn_dct.keys())
        for item in corpus:
            vec=[0]*no_of_terms                                 # Empty vector of terms for a document
            vector = trn_model[item]                            # Entropy based vectors
            for elm in vector:
                vec[elm[0]]=elm[1]
            trn_vec.append(vec)
    # Classificiation and feature selection pipelines
        clf,clf_parameters,ext2=self.classification_pipeline()
        if self.no_of_selected_terms==None:                                  # To use all the terms of the vocabulary
            pipeline = Pipeline([('clf', clf),])    
        else:
            try: 
                pipeline = Pipeline([('feature_selection', SelectKBest(chi2, k=self.no_of_selected_terms)), 
                    ('clf', clf),])  
            except:                                  # If the input is wrong
                print('Wrong Input. Enter number of terms correctly. \n')
                sys.exit()
        grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10) 
        grid.fit(trn_vec,trn_cat)     
        clf= grid.best_estimator_
        print(clf)
        return clf,ext2,trn_dct,trn_model

# BERT model accuracy function
     def compute_metrics(self,pred):
         labels = pred.label_ids
         preds = pred.predictions.argmax(-1)
         acc = accuracy_score(labels, preds)
         return {
             'accuracy': acc,
         }     

# BERT model    
     def bert_model(self,trn_data,trn_cat,test_size=0.2,max_length=512): 
        print('\n ***** Running BERT Model ***** \n')       
        tokenizer = BertTokenizerFast.from_pretrained(self.model_source, do_lower_case=True) 
        labels=np.asarray(trn_cat)     # Class labels in nparray format     

        (train_texts, valid_texts, train_labels, valid_labels), class_names = train_test_split(trn_data, labels, test_size=test_size), trn_cat
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
        train_dataset = get_torch_data_format(train_encodings, train_labels)
        valid_dataset = get_torch_data_format(valid_encodings, valid_labels)
        model = BertForSequenceClassification.from_pretrained(self.model_source, num_labels=len(class_names)).to("cpu")
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=20,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            logging_steps=200,               # log & save weights each logging_steps
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            )    
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
            )
        print('\n Trainer done \n')
        trainer.train()
        print('\n Trainer train done \n')        
        trainer.evaluate()
        print('\n save model \n')
        model_path = self.path+"bert_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        return model,tokenizer,class_names

# Classification using the Gold Statndard after creating it from the raw text    
     def sentence_classification(self):
        if self.model=='doc2vec':
            ln = input("Enter the length of the Doc2Vec word vector: \t")
        trn_data=[];    trn_cat=[];   
        p1=0; p2=0;
    
        fp=open(self.path+'training_relevant_class_data.csv',"r")
        fn=open(self.path+'training_irrelevant_class_data.csv',"r")    
        rel_sent = list(csv.reader(fp,delimiter='\n')) 
        irl_sent = list(csv.reader(fn,delimiter='\n'))

        print('\n ***** Processing Training Documents ***** \n')
    # Getting Relevant Sentences of Training Corpus
        for item in rel_sent:
            text=''.join(item)
            if text.split(',')[1]:
                text=text.split(',')[1]
                text=re.sub(r'\n', '', text)                            # Remove \n
                sentences = tokenize.sent_tokenize(text)
                for sentence in sentences:
#                    sentence=' '.join([word for word in sentence.lower().rstrip('.').split(' ') if word not in en_stopwords]) # Stopword removal                    
                    sentence=re.sub(r'\d+\.\d+', '', sentence)          # Remove floating point numbers
                    sentence.rstrip('.|?|\)|\]|\'|\"|;|`')
                    trn_data.append(sentence)           
                    trn_cat.append(0)
                    p1=p1+1      
    # Getting Irrelevant Sentences of Training Corpus        
        for item in irl_sent:
            text=''.join(item)
            if text.split(',')[1]:
                text=re.sub(r'\n', '', text)                            # Remove \n
                text=text.split(',')[1]
                sentences = tokenize.sent_tokenize(text)
                for sentence in sentences:
                    sentence.rstrip('.|?|\)|\]|\'|\"|;|`')
                    sentence=' '.join([word for word in sentence.lower().rstrip('.').split(' ') if word not in en_stopwords]) # Stopword removal                                        
                    sentence=re.sub(r'\d+\.\d+', '', sentence)          # Remove floating point numbers
                    trn_data.append(sentence)           
                    trn_cat.append(1)
                    p2=p2+1 
 # Processing Test Samples
        if not os.path.isdir(self.path+'test_data'):
            print('The directory of test_data does not exist \n')
            sys.exit(0)
        else:
            tst_files=os.listdir(self.path+'test_data')
        count=0; p3=0; 
        if tst_files==[]:
            print('There is no test samples in the directory \n')
        else:
 # Calling the training model
            print('\n ***** Building Training Model ***** \n')
            if self.model=='tfidf':
                clf,ext2=self.tfidf_training_model(trn_data,trn_cat)
            elif self.model=='entropy':
                clf,ext2,trn_dct,trn_model=self.entropy_training_model(trn_data,trn_cat)
            elif self.model=='doc2vec':
                clf,ext2,trn_model=self.doc2vec_training_model(ln,trn_data,trn_cat)
            elif self.model=='bert':
                trn_model,trn_tokenizer,class_names=self.bert_model(trn_data,trn_cat)                
            print('\n ***** Processing Test Documents ***** \n')
            for item in tst_files:
                if item.find('.pdf')>0:             # Checking if it is a PDF file 
                    count+=1
                    tst_data=[]; tst_data_cleaned=[]; tst_vec=[]; tst_sentences=[] 
                    print(item.lower().rstrip('.pdf'))
                    out = open(self.path+'output/'+item.lower().rstrip('.pdf')+'.txt',"w")   # Output file 
                    text=self.pdf_to_text(self.path+'test_data/'+item) 
                    text=self.text_refinement(text)                     # Cleaning text file
                    sentences = tokenize.sent_tokenize(text)
                    for sentence in sentences:                          # Extracting sentences
                        tst_data.append(sentence)  
                        sentence=' '.join([word for word in sentence.lower().rstrip('.').split(' ') if word not in en_stopwords])
                        tst_data_cleaned.append(sentence)                        
                        p3=p3+1
    # Classification of the test samples 
                    if self.model=='tfidf':
                        out.write('\n Using '+ext2+' Classifier: \n\n') 
                        predicted = clf.predict(tst_data_cleaned) 
                    elif self.model=='entropy':
                        out.write('\n Using '+ext2+' Classifier: \n\n') 
                        for sentence in tst_data_cleaned:
                            sentence=nltk.word_tokenize(sentence.lower()) 
                            tst_sentences.append(sentence)                                
                        corpus = [trn_dct.doc2bow(row) for row in tst_sentences]     
                        no_of_terms=len(trn_dct.keys())
                        for itm in corpus:
                            vec=[0]*no_of_terms                          # Empty vector of terms for a document
                            vector = trn_model[itm]                      # LogEntropy Vectors 
                            for elm in vector:
                                vec[elm[0]]=elm[1]
                            tst_vec.append(vec)        
                        predicted = clf.predict(tst_vec) 
                    elif self.model=='doc2vec':
                        out.write('\n Using '+ext2+' Classifier: \n\n') 
                        for sentence in tst_data_cleaned:
                            sentence=nltk.word_tokenize(sentence.lower())
                            inf_vec = trn_model.infer_vector(sentence,epochs=100)
                            tst_vec.append(inf_vec)
                        predicted = clf.predict(tst_vec)
                    elif self.model=='bert':
                        out.write('\n Using BERT Model: \n\n') 
                        predicted=[]
                        for sentence in tst_data_cleaned:
                            inputs = trn_tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cpu") 
                            outputs = trn_model(**inputs)
                            probs = outputs[0].softmax(1)
                            cl=class_names[probs.argmax()]
                            predicted.append(cl)                            
                    elif self.model=='keyword_matching':
                        out.write('\n Following Keyword Match \n\n') 
                        predicted=self.keyword_matching_model(tst_data_cleaned)
                          
                    out.write('Total No. of Sentences in Test Sample: '+str(p3)+'\n\n')
                    out.write('The relevant sentences are as follow: \n')
                    nps=0
                    for i in range(0,len(predicted)):
         # Check if there is a floating point number                 
                        if predicted[i] == 0 and re.findall(r'\d+\.\d+', tst_data[i])!=[]:
                            nps=nps+1                 
                            tst_data[i]=re.sub(r'(\d+\.\d+°*)(\s*\�\s*)(\d+\.\d+)', r'\1 ± \3',tst_data[i]) # SHIFT+OPTION+PLUS Sign - plus-minus symbol
                            out.write('\n'+str(nps)+")  "+tst_data[i]+'\n')               
                    print("Total No. of Relevant Sentences of "+item+" : %d\n" %nps)
            
            print('No of sentences belong to RELEVANT class of the training corpus: '+ str(p1)) 
            print('No of sentences belong to IRRELEVANT class of the training corpus: '+ str(p2)) 
            print('No of sentences belong to the TEST corpus: '+ str(p3)) 
        
    
        
