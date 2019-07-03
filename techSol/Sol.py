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

#reading csv file as a panda dataframe
df = pd.read_csv('train.tsv', sep = '\t')
# create a random array of same length as the csv file
msk = np.random.rand(len(df)) < 0.8
# create a Train array from all True values from msk
train = df[msk]
# create a Train array from all False values from msk
test = df[~msk]

# ploting
plt.subplot(1, 2, 1)
(train['price']).plot.hist(bins=50, 
edgecolor = 'white', range = [0, 250])
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.subplot(1, 2, 2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(9,4),
edgecolor='white')
plt.xlabel('log(price+1)', fontsize=12)
plt.title('Price Distribution', fontsize=12)

#counting the values of the shippings shows that 55% of the items where
#paid by the buyers
train['shipping'].value_counts() / len(train)

# How does the shippment and the price compare?
shipping_fee_by_buyer = train.loc[df['shipping'] == 0, 'price']
shipping_fee_by_seller = train.loc[df['shipping'] == 1, 'price']

#plot the difference
fig, ax = plt.subplots()
ax.hist(shipping_fee_by_seller,bins=100,range=[0,100],
        label='Price when the seller pays for shipping')
ax.hist(shipping_fee_by_buyer,bins=100,range=[0,100],
        label='Price when the buyer pays for shipping')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price by shipping type')
plt.tick_params(labelsize=12)
plt.legend()
plt.show()

print('The average when the seller is paying is {}'.format(round(shipping_fee_by_seller.mean(),2)))
print('The average when the buyer is paying is {}'.format(round(shipping_fee_by_buyer.mean(),2)))

# how many unique categories are there?
print('There are {}'.format(train['category_name'].nunique()),' unique categories')

train['category_name'].value_counts()[:10]

#how does the condition and the price compare?
sns.boxplot(x = 'item_condition_id',y = np.log(train['price']+1),
            data=train,palette=sns.color_palette('RdBu',5))
