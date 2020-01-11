
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib_venn import venn2

import xml.etree.ElementTree as ET
import json 

from sklearn.feature_extraction.text import TfidfVectorizer


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


class Feature_Extraction(object):
    '''
    Helper class used for Feature engineering purposes.
    
    '''
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def unconstrained_chars(self):
        pass

    def shortcuts_removal(self):
        
        pass

    def lower_case(self):
        self.dataframe[column] = self.dataframe[column].str.lower()



    