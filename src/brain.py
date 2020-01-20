
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from matplotlib_venn import venn2


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer

import json
import requests

# import tensorflow_hub as hub
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.linear_model import MultiTaskElasticNet
import tensorflow as tf
import torch
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM


class Parser(object):
    '''
    Parse config.xml file.

    config.xml file stores hyperparameters used by machine learning algorithms.

    It allows to make changes quickly without seeking specific values in code.
    '''

    def __init__(self):
        self.root = ET.parse('src/config.xml').getroot()
    
    
#int(self.root[0][0].text)

class Analysis(object):
    '''
    Store code which is referenced by agent.ipynb.

    It helps to get rid of redundant code in jupyter notebook, and
    also makes it more cleaner.
    '''

    def host_distribution(self, distribution, title):
        fig = px.pie(names=distribution.index, 
             values=distribution.values, 
             title=title,
             width=800, 
             height=800)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
        fig.show()
        
    def categories_distribution(self, categories, title):
        
        category_share = pd.DataFrame({'share': categories.value_counts() / categories.count()})
        category_share['category'] = category_share.index   
        
        fig = px.bar(category_share, x='category', y='share',
            labels={'share':'share in %'},
            title=title)
        
        fig.show()
    
    def venn_diagrams(self, columns, plot_num, title):
        plt.subplot(plot_num)
        venn2([set(columns[0].unique()), set(columns[1].unique())], set_labels = ('Train set', 'Test set'))
        plt.title(title)
        plt.show()
    
    def distribution_imposition(self, first_col, second_col, title):
        plt.figure(figsize=(20, 6))
        sns.distplot(first_col.str.len())
        sns.distplot(second_col.str.len())
        plt.title(title)
        plt.show()

class Feature_Engineering(object):
    '''
    Helper class used for Feature engineering purposes.
    
    '''
    def __init__(self, dataframe):
        self.dataframe = dataframe
        with open("src/charset.json", encoding="utf8") as json_file:
            self.charset  = json.load(json_file)
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        # change number of compoments
        self.tsvd = TruncatedSVD(n_components = 128, n_iter=5)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.binarizer = LabelBinarizer()


    def unconstrained_chars(self, column, index):
        df = self.dataframe[column][index]
        for char in self.charset['CHARS']:
            df = df.replace(char, '')
        return df

    def shortcuts_removal(self, column, index):
        return ' '.join(list(map(lambda word: self.find_and_replace(word), self.dataframe[column][index].split())))
        
    def lower_case(self, column):
        return self.dataframe[column].str.lower()
        
    
    def find_and_replace(self, word):
        for key, value in self.charset['SHORTCUTS'].items():
            if key == word:
                return value
        return word

    def flow(self, column):
        self.dataframe[column] = self.lower_case(column)
        for index in range(self.dataframe[column].shape[0]):
            self.dataframe[column][index] = self.unconstrained_chars(column, index)
            self.dataframe[column][index] = self.shortcuts_removal(column, index)

        return self.dataframe[column]
    
    def tfidf_vec(self, column):
        return list(self.tsvd.fit_transform(self.vectorizer.fit_transform(self.dataframe[column].values)))
    
    def binarize(self, column):
        return list(self.binarizer.fit_transform(self.dataframe[column].values))
    
    def bert_separators(self, column):
        for index in range(self.dataframe[column].shape[0]):
            self.dataframe[column][index] = self.dataframe[column][index].split('.')
            
        return self.dataframe[column]
    
    def model_conf(self):
        self.model.cpu()
        
    
    def make_vectors(self, column):
        ids = self.dataframe[column].str.slice(0, 500).apply(self.tokenizer.encode)
        vectors = []

        for column in tqdm(ids):
            input_ids = torch.Tensor(column).to(torch.int64).unsqueeze(0)
            try:
                outputs = self.model(input_ids.cpu())
                vectors.append(outputs[0].detach().cpu().numpy().max(axis = 1))

            except:
                vectors.append(np.zeros(outputs[0].detach().cpu().numpy().max(axis = 1)).shape)
        
        return vectors

    