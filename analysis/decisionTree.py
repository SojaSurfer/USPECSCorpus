from pathlib import Path
import sys
import string

import sklearn.linear_model
import sklearn.tree
import sklearn.dummy
from sklearn.model_selection import train_test_split
from nltk import FreqDist
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from tqdm import tqdm
import graphviz
from analysis import loadMetadata, concatTables



STOPWORDS = stopwords.words() + list(string.punctuation) + ["'s", "--", 'applause', "’re", "--ms", "--the"]


def getWordFrequencySpeaker(amount: int = 10) -> pd.DataFrame:
    metadataDF: pd.DataFrame = loadMetadata()
    totalDF = pd.DataFrame()

    root = Path('corpus') / 'tables'

    for index, row in tqdm(metadataDF.iterrows(), ncols=80, total=len(metadataDF)):

        tablePath = root / row['linkTables']

        tableDF = pd.read_csv(tablePath)


        fdist = FreqDist(cleanTokens(tableDF))
        mostCommon = fdist.most_common(amount)
        
        df = pd.DataFrame(data=dict(mostCommon) | {'PERIOD': row['period'], 'SPEAKER': row['speaker']}, index=[index])        

        totalDF = totalDF.combine_first(df)


    totalDF.fillna(0, inplace=True)
    totalDF = totalDF.convert_dtypes()

    return totalDF


def cleanTokens(tableDF:pd.DataFrame) -> list:
    return [w.lower() for w in tableDF['LEMMA'] if not w in STOPWORDS and w.isalpha()]


def renderDecisionTree(tree, y) -> None:
    sklearn.tree.export_graphviz(tree, out_file='decisiontree.dot', 
                    feature_names=df.columns, 
                    class_names=y.unique(), 
                    filled=True, rounded=True, 
                    special_characters=True)

    with open('decisiontree.dot') as f:
        dot_graph = f.read()

    graph = graphviz.Source(dot_graph)
    graph.render('decisionTree', format='png', cleanup=True)

    return None


def trainTreeClassifier(df:pd.DataFrame) -> sklearn.tree.DecisionTreeClassifier:
    return treeClf



if __name__ == '__main__':
    # totalDF = getWordFrequencySpeaker()
    # totalDF.to_parquet('wordFreq.parquet', index=False)

    df = pd.read_parquet('wordFreq.parquet')
    df = df.convert_dtypes()
    df.pop('PERIOD')

    y = df.pop('SPEAKER')
    train, test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    treeClf = sklearn.tree.DecisionTreeClassifier(max_depth=9)

    treeClf.fit(train, y_train)

    accuracy = treeClf.score(test, y_test)
    print(f'Accuracy: {accuracy:.2f}')

    for strategy in ['most_frequent', 'prior', 'stratified', 'uniform']:
        dummyClf = sklearn.dummy.DummyClassifier(strategy=strategy)
        dummyClf.fit(train, y_train)
        accuracy = dummyClf.score(test, y_test)
        print(f'Accuracy Dummy: {accuracy:.2f}')

    # renderDecisionTree(treeClf, y)



