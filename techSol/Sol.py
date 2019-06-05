import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
#import lightgbm as lgb

# for better numbers visibility
pd.set_option('float_format', '{:f}'.format)

#reading csv file
df = pd.read_csv('train.tsv', sep = '\t')
# create a random array of same length as the csv file
msk = np.random.rand(len(df)) < 0.8
# create a Train array from all True values from msk
train = df[msk]
# create a Train array from all False values from msk
test = df[~msk]

train.price.describe()