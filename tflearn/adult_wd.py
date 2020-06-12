#
# tensorflow 原装wide & deep
# census 数据预测收入
#
import os
import urllib.request
from tflearn import utils
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras

COLUMNS = [('age', tf.int32), 
           ('workclass', tf.string), 
           ('fnlwgt', tf.int32), 
           ('education', tf.string),
           ('education_num', tf.int32),
           ('marital_status', tf.string),
           ('occupation', tf.string),
           ('relationship', tf.string),
           ('race', tf.string), 
           ('gender', tf.string),
           ('capital_gain', tf.int32),
           ('capital_loss', tf.int32),
           ('hours_per_week', tf.int32),
           ('native_country', tf.string)
           ]
 

def load_data():
    return utils.load_adult_data()

def build_input_layer():
    '''输入层
    '''
    schema = {}
    for name, t in COLUMNS:
        schema[name] = keras.Input(shape=(1,), name=name, dtype=t)
    return schema

def build_feature_columns():
    age = feature_column.numeric_column('age')
    workclass = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('workclass',
                ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']))
    fnlwgt = feature_column.numeric_column('fnlwgt')
    education = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('education',
                ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']))
    education_num = feature_column.numeric_column('education_num')
    marital_status = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('marital_status',
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                 'Married-spouse-absent', 'Married-AF-spouse']))
    occupation = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('occupation',
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
                 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                 'Armed-Forces']))
    relationship = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('relationship',
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']))
    race = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('race', 
                ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']))
    gender = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('gender', 
                ['Female', 'Male']))    
    capital_gain = feature_column.numeric_column('capital_gain') 
    capital_loss = feature_column.numeric_column('capital_loss')
    hours_per_week = feature_column.numeric_column('hours_per_week')
    native_country = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list('native_country',
                ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
                 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 
                 'Holand-Netherlands']))
    wide = [age, workclass, race, gender, capital_gain, capital_loss, hours_per_week, native_country]    
    deep = [age, workclass, fnlwgt, education, education_num, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]
    return (wide, deep)

def build_model_wide():
    w, d = build_feature_columns()
    feature_layer = keras.layers.DenseFeatures(w)
    model = keras.Sequential([
        feature_layer,
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

def build_model_deep():
    w, d = build_feature_columns()
    feature_layer = keras.layers.DenseFeatures(d)
    model = keras.Sequential([
        feature_layer,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC()])
    return model


def build_model_wide_deep():
    '''functional api
    '''
    w, d = build_feature_columns()
    input_layer = build_input_layer()
    feature_layer = keras.layers.DenseFeatures(w)(input_layer)
    hidden_layer = keras.layers.Dense(128, activation="relu")(feature_layer)
    output_layer = keras.layers.Dense(1)(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

if __name__ == '__main__':
    utils.init_logging()
    #model = build_model_wide()
    model = build_model_deep()
    #model = build_model_wide_deep()
    train_ds, test_ds = load_data()
    model.fit(train_ds, epochs=10)
    loss, accuracy, auc = model.evaluate(test_ds)
    print("Accuracy", accuracy, "auc", auc)
    #for x, y in test_ds: 
    #    y_pred = model.predict(x)
    #    for v, v1 in zip(y, y_pred):
    #        print(v.as_numpy(), v1)