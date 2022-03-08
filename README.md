# A Sentence Classification Framework to Identify Geometric Errors of Radiotherapy from Literature
The aim of this project is to build a framework to extract sentences from relevant literature of radiotherapy. The framework builds a training corpus by extracting sentences containing different data elements of geometric errors of radiotherapy from relevant publications by using some keywords fixed by the domain expert. Subsequently, the method trains a machine learning classifier e.g., Support Vector Machine using the training corpus to extract the sentences containing desired geometric errors from test documents. The experiments are conducted on 60 publications to automatically extract the sentences containing geometric errors of radiotherapy.  Read the [paper](https://www.mdpi.com/2078-2489/12/4/139/htm) for more information.

## Prerequisites
The following libraries have to be installed one by one before running the code, if they are not already installed.

[Fitz](https://pypi.org/project/fitz/), [PyMuPDF](https://pypi.org/project/PyMuPDF/), [Gensim](https://github.com/RaRe-Technologies/gensim), [NLTK](https://www.nltk.org/install.html), [NumPy](https://numpy.org/install/), [Python 3.7 or later version](https://www.python.org/downloads/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)

## How to run the framework?

Pass the path of the project e.g., `/home/xyz/sentence_classification/` as a parameter of the main class in `sentence_classification.py`. Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore keep the individual PDFs of training and test data in the respective directories. The list of keywords to build the training data should be stored as `keywords.txt` in the main project path. Create a directory, called, `output` in the main project path to store the outputs of individual test samples. 

Subsequently, run the following lines to get relevant sentences of geometric errors of radiotherapy for individual test documents. 

```
de=data_extraction('/home/xyz/sentence_classification/',model='entropy',clf_opt='s',no_of_selected_terms=1500,threshold=0.5)  
de.build_training_data()       
de.sentence_classification()
```

The following options of `model` are available and the `default` is `entropy`: 

        'bert' for BERT model

        'entropy' for Entropy based term weighting scheme

        'doc2vec' for Doc2Vec based embeddings 

        'tfidf' for TF-IDF based term weighting scheme 

The following options of 'clf_opt' are available and the `default` is `s`: 

        'lr' for Logistic Regression 

        'ls' for Linear SVC

        'n' for Multinomial Naive Bayes

        'r' for Random Forest

        's' for Support Vector Machine 

`model_source` is the path of BERT model from [Hugging Face](https://huggingface.co/models?search=biobert) or from the local drive. The default option is `monologg/biobert_v1.1_pubmed`. `vec_len` is the desired length of the feature vectors developed by the Doc2Vec model. The deafult option of `no_of_selected_terms` is `None`, otherwise desired number of terms should be mentioned. The default option of threshold (i.e., the sentence similarity threshold α) is 0.5. An example code to implement the whole model is uploaded as `testing_data_extraction.py`. 

### Note
The required portion of the code (in `sentence_classification.py`) to run a given BERT model is commented, as in many standalone machine one may face difficulty in installing BERT. These comments has to be removed in order to run BERT. Otherwise, a separate code (`sentence_classification_bert.py`) is given to run BERT model for the given data on a standalone machine. 

## Contact

For any further query, comment or suggestion, you may reach out to me at welcometanmay@gmail.com

## Citation
```
@article{basu21,
  title={A Sentence Classification Framework to Identify Geometric Errors in Radiation Therapy from Relevant Literature},
  author={Basu, Tanmay and Goldsworthy, Simon and Gkoutos, Georgios V.},
  journal={Information},
  volume={12},
  number={4},
  pages={139},
  year={2021},
  publisher={MDPI, Switzerland}
}
```
