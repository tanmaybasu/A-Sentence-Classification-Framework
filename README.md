# A Sentence Classification Framework to Identify Geometric Errors of Radiotherapy from Literature
The aim of this project is to build a framework using machine learning to extract required data elements of geometric errors of radiotherapy from relevant literaure. The framework builds a training corpus by extracting senetnces containing different data elements of geometric errors of radiotherapy from relevant publications by using some keywords fixed by the domain expert. The articles are retrieved from PubMED following a given set of rules by the domain expert. Subsequently, the method trains a machine learning classifier e.g., Support Vector Machine using this training corpus to extract the sentences containing desired geometric errors from test documents. The experiments are conducted on 60 publications to automatically extract the sentences containing geometric errors of radiotherapy.  

## Prerequsites
[Python 3](https://www.python.org/downloads/), [Scikit-Learn](https://scikit-learn.org/0.16/install.html), [NLTK](https://www.nltk.org/install.html), [Fitz](https://pypi.org/project/fitz/), [Gensim](https://github.com/RaRe-Technologies/gensim)

## How to run the framework?

Pass the path of the project e.g., `/home/xyz/data_extraction/` as a parameter of the main class. Create the following directories inside this path: 1) `training_data`, 2) `test_data`. Therefore keep the individual PDFs of training and test data in the respective directories. The list of keywords to build the training data should be stored as `keywords.txt` in the main project path. Create a directory, called, `output` in the main project path to store the outputs of individual test samples. 

Subsequently, run the following lines to get relevant sentences of anaxiety outcome measures for individual test samples. 

```
de=data_extraction('/Users/basut/geometric_error_extraction/code/',model='tfidf',clf_opt='s',no_of_selected_terms=1000)  
de.build_training_data()       
de.sentence_classification()
```

Here model indicates the feature weighting scheme and the options are 'tfidf' for tf-idf based scheme, 'entropy' for entropy based weighting, 'doc2vec' for doc2vec based embeddings and default is 'tfidf'. The options of 'clf_opt' are 'lr' for logistic regression, 'ls' for linear svc, 'n' for naive bayes, 'r' for random forest, 's' for support vector machine and default is 's'. The deafult option of no_of_selected_terms is None, otherwise desired number of terms is needed. An example code to implement the whole model is uploaded as `testing_data_extraction.py`. 

## Contact

For any further query, comment or suggestion, you may reach out to me at welcometanmay [at] gmail.com

## Citation
```

```
