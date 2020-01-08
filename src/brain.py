import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer


class Parser(object):
    '''
    Parse config.xml file.

    config.xml file stores hyperparameters used by machine learning algorithms
    and analysis calibration.

    It allows to make changes quickly without seeking specific values in code.
    '''

    def __init__(self):
        self.root = ET.parse('src/config.xml').getroot()
    
    


class Analysis(Parser):
    '''
    Store code which is referenced by agent.ipynb.

    It helps to get rid of redundant code in jupyter notebook, and
    also makes it more cleaner.
    '''

    def host_distribution(self, distribution, title):
        fig = px.pie(names=distribution.index, 
             values=distribution.values, 
             title=title,
             width=int(self.root[0][0].text), height=int(self.root[0][1].text))

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
        