#
# tensorflow 原装wide & deep
# census 数据预测收入
#
import os
import urllib.request
from tflearn import utils
from tensorflow import feature_column
from tensorflow import keras


def load_data():
    return utils.load_adult_data()

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
    deep = []
    return (wide, deep)

def build_model():
    w, d = build_feature_columns()
    feature_layer = keras.layers.DenseFeatures(w) 
    model = keras.Sequential([
        feature_layer,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    print(model.summary())
    return model

def build_model_v2():
    '''functional api
    '''
    w, d = build_feature_columns()
    feature_layer = keras.layers.DenseFeatures(w)
    hidden = keras.layers.Dense(128, activation="relu")(feature_layer)
    output = keras.layers.Dense(1)(hidden)
    model = keras.models.Model(inputs=feature_layer, output=output)
    print(model.summary())
    model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

if __name__ == '__main__':
    utils.init_logging()
    model = build_model_v2()
    #train_ds, test_ds = load_data()
    #model.fit(train_ds,
    #      epochs=10)

    #loss, accuracy = model.evaluate(test_ds)
    #print("Accuracy", accuracy, "error", 1-accuracy)
