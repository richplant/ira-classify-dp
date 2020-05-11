from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
import numpy as np
import spacy
import torch
from spacy.compat import cupy


class EmbedTransformer(TransformerMixin):
    """
    Generate embeddings with pre-trained BERT
    """
    def __init__(self):
        is_using_gpu = spacy.require_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.nlp = spacy.load('en_trf_bertbaseuncased_lg')

    def fit_transform(self, data, **kwargs):
        new_x = []
        for doc in self.nlp.pipe(data):
            if len(doc) > 0:
                new_x.append(cupy.asnumpy(doc[0].vector))
            else:
                new_x.append(np.zeros(768))
        return np.array(new_x)


class PreProcessor:
    """
    Perform pre-processing
    Import dataframe and apply sklearn transformations
    Vectorize (non full text) strings with TF/IDF
    Normalise int values, remove mean and scale to unit variance
    Instantiate an embedding transformer for message text features
    """
    def __init__(self):
        self.mapper = DataFrameMapper([
            (['created_at'], StandardScaler()),
            (['user_created_at'], StandardScaler()),
            (['favorite_count'], StandardScaler()),
            (['retweet_count'], StandardScaler()),
            (['user_followers_count'], StandardScaler()),
            (['user_following_count'], StandardScaler()),
            ('hashtags', TfidfVectorizer(max_features=1_000)),
            ('urls', TfidfVectorizer(max_features=1_000)),
            ('user_description', EmbedTransformer()),
            ('user_location', TfidfVectorizer(max_features=1_000)),
            ('user_name', TfidfVectorizer(max_features=1_000)),
            ('user_screen_name', TfidfVectorizer(max_features=1_000)),
            ('user_profile_urls', TfidfVectorizer(max_features=1_000)),
            ('full_text', EmbedTransformer())
        ])

    def transform(self, df):
        labels = label_binarize(df.pop('label'), classes=['none', 'astroturf'])
        return self.mapper.fit_transform(df), labels