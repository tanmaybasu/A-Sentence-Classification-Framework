#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:10:52 2021

@author: basut
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:11:07 2021

@author: Tanmay Basu
"""

import csv,fitz,os,re,sys,torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from nltk import tokenize
import nltk
from nltk.corpus import stopwords
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# max sequence length for each document/sentence sample
max_length = 512
model_name = "monologg/biobert_v1.1_pubmed"

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True) 
path='/Users/basut/geometric_error_extraction/code/'
   
# PDF to text conversion
def pdf_to_text(file):        
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
def text_refinement(text):
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
def get_sent_score(sent,phrase,inp):
    sent_tokens = nltk.word_tokenize(sent.lower().strip('.'))
    sent_tokens = [item.rstrip('s') for item in sent_tokens]            # Converting the plurals to singulars  
    target_tokens = nltk.word_tokenize(phrase.lower().strip('.'))              
    target_tokens = [item.rstrip('s') for item in target_tokens]        # Converting the plurals to singulars             
    score = 0   
#   Using modified Jaccard Similarity    
    if inp=='0':                                                        # Jaccard is the default similarity measure
        for token in target_tokens:
            if token in sent_tokens and token not in stopwords.words('english'):    # Discarding the stopwords 
                score += 1
        if score!=0:
            score = float(score)/len(target_tokens)       
    return score

# Check if a sentence is relevant to the data element 
def check_relevant_sentences(sentence,keywords):
    phrase_score=[]; total_score=0.0
    sentence = re.sub(r'[^a-zA-Z0-9.?:!$\n]', ' ', sentence)    # Remove special character 
    for phrase in keywords:
        score=0.0;
        score=get_sent_score(sentence,phrase,'0')
        total_score+=score
        if score>=0.5:
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
def build_training_data():
     if os.path.isfile(path+'keywords.txt') and os.path.getsize(path+'keywords.txt') > 0:                      
         fk=open(path+'keywords.txt', "r") 
         keywords = list(csv.reader(fk,delimiter='\n'))
         keywords = [item for sublist in keywords for item in sublist]
         fk.close()
     else:                                                  # If the keywords file does not exist or is empty then exit 
         print('Either keywords.txt does not exist or it is empty \n')
         sys.exit(0)
# If the training data directory does not exist then exit
     if not os.path.isdir(path+'training_data'):
        print('The directory training_data does not exist \n')
        sys.exit(0)
     trn_files=os.listdir(path+'training_data')
     if trn_files==[]:
         print('There is no training samples in the directory \n')
     else:
         count=0;
         print('############### Preparing Training Data ############### \n')
         fp=open(path+'training_relevant_class_data.csv',"w")
         fn=open(path+'training_irrelevant_class_data.csv',"w")  
         for item in trn_files:
            if item.find('.pdf')>0:                       # Checking if it is a PDF file
                count+=1
                text=pdf_to_text(path+'training_data/'+item) 
                text=text_refinement(text)                     # Cleaning text file 
                fp.write(item.rstrip('.pdf')+',')
                fn.write(item.rstrip('.pdf')+',')
                if text!='':
                    text=re.sub(r',', r'', text)      # replacing , by ; to build the csv properly 
                    text=re.sub(r'\n', r'\\n ', text)    # replacing \n by '\\n ' to build the csv properly  
                    sentences=nltk.sent_tokenize(text)
                    for sentence in sentences:
                        if sentence!='':
                            phrase_score=check_relevant_sentences(sentence,keywords)
                            if phrase_score!=[]:
                                ln=len(nltk.word_tokenize(phrase_score[0][0]))
         # Check if there is a floating point number 
                                if ln>2 and phrase_score[0][1]>=0.5 and re.findall(r'\d+\.\d+', sentence)!=[]:    
                                    fp.write(sentence+' ')
                                elif ln<=2 and phrase_score[0][1]>0.5 and re.findall(r'\d+\.\d+', sentence)!=[]: 
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

def get_training_data(test_size=0.2):
    documents=[]; class_labels=[]; 
    f1=open(path+'training_relevant_class_data.csv', 'r')
    reader=list(csv.reader(f1,delimiter='\n'))
    f1.close()
    for item in reader:
        text=''.join(item)
        documents.append(text)
        class_labels.append(0)
    f2=open(path+'training_irrelevant_class_data.csv', 'r')
    reader=list(csv.reader(f2,delimiter='\n'))
    f2.close()
    for item in reader:
        text=''.join(item)
        documents.append(text)
        class_labels.append(1)
    labels=np.asarray(class_labels)     # Class labels in nparray format 
    return train_test_split(documents, labels, test_size=test_size), class_labels
  
# call the function
(train_texts, valid_texts, train_labels, valid_labels), target_names = get_training_data()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

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

# convert our tokenized data into a torch Dataset
train_dataset = get_torch_data_format(train_encodings, train_labels)
valid_dataset = get_torch_data_format(valid_encodings, valid_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cpu")

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
  
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()

model_path = "bert_model_geometric_error"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

def get_prediction(path):

    # Processing Test Samples
    if not os.path.isdir(path+'test_data'):
        print('The directory of test_data does not exist \n')
        sys.exit(0)
    else:
        tst_files=os.listdir(path+'test_data')
    count=0; p1=0; p2=0; p3=0; 
    if tst_files==[]:
        print('There is no test samples in the directory \n')
    else:
        print('\n ***** Processing Test Samples ***** \n')
        for item in tst_files:
            if item.find('.pdf')>0:             # Checking if it is a PDF file 
                count+=1
                tst_data=[]; predicted=[]; 
                print(item.lower().rstrip('.pdf'))
                out = open(path+'output/'+item.lower().rstrip('.pdf')+'.txt',"w")   # Output file 
                text=pdf_to_text(path+'test_data/'+item) 
#                text=text_refinement(text)                     # Cleaning text file
                sentences = tokenize.sent_tokenize(text)
                for sentence in sentences:                          # Extracting sentences
                    tst_data.append(sentence)   
                    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cpu") 
                    outputs = model(**inputs)
                    probs = outputs[0].softmax(1)
                    cl=target_names[probs.argmax()]
                    predicted.append(cl)
                    p3=p3+1
                out.write('\n Using BERT Classifier: \n\n')                                
                out.write('Total No. of Sentences in Test Sample: '+str(p3)+'\n\n')
                out.write('The relevant sentences are as follow: \n')
                nps=0
#                print(predicted)
#                input()
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
    
get_prediction('/Users/basut/geometric_error_extraction/code/')


