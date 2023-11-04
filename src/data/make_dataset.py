# -*- coding: utf-8 -*-
# Doing all the necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# loading the data into a pandas dataframe
df = pd.read_csv('data/interim/filtered.tsv', sep='\t')

# Taking correctly paraphrazed toxic texts
df_filtered = df.sort_values(by='trn_tox', ascending=True).head(100000)
df_filtered.reset_index(drop=True, inplace=True)

# Create the train and test datasets
train_test_dataset = df_filtered[['reference', 'translation']]

# Split the dataset into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(train_test_dataset, test_size=0.2, random_state=42)

train.to_csv('data/interim/train.txt', sep = '\n', index = False)
test.to_csv('data/interim/test.csv', index = False)

"""## Lets create a dataset for fine-tuning GPT-2"""

train['combined'] = '<s>' + train.reference + '</s>' + '>>>>' + '<p>' + train.translation + '</p>'
train.combined.to_csv('combined.txt', sep = '\n', index = False)

