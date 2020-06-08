#
# 工具模块
#
import os
import sys
import urllib.request
import urllib.parse
import logging
import tensorflow as tf
import numpy as np
import pandas as pd

def init_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(message)s',
        level=logging.INFO,
    )


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format."""
    temp_file, _ = urllib.request.urlretrieve(url)
    with open(temp_file, 'r') as temp_eval_file:
        with open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                eval_file.write(line)
    os.remove(temp_file)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def _df_to_dataset(dataframe, label_column='label', shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    label = dataframe.pop(label_column)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), label))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def _do_load_adult_data(url):
    cache_file_path = './data'+os.path.basename(urllib.parse.urlparse(url).path) 

    if not os.path.exists(cache_file_path):
        logging.info("downloading %s", cache_file_path)
        _download_and_clean_file(cache_file_path, url)
 
    COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket']
    
    df = pd.read_csv(cache_file_path, names=COLUMN_NAMES) 
    df['label'] = df.pop('income_bracket').transform(lambda x: 1 if x == '>50K' else 0)
    ds = _df_to_dataset(df)
    return ds

def load_adult_data():
    '''美国统计局人口调查数据，用于预测收入。
       来自tensorflow wider&deep example
    '''
    DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
    TRAINING_FILE = 'adult.data'
    TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
    TEST_FILE = 'adult.test'
    TEST_URL = '%s/%s' % (DATA_URL, TEST_FILE)
   
    return map(_do_load_adult_data, [TRAINING_URL, TEST_URL]) 
    