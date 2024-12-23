from typing import Iterable
from pathlib import Path
from time import perf_counter as pc
import concurrent.futures

import pandas as pd



def loadMetadata() -> pd.DataFrame:
    path = Path('corpus') / 'metadata.xlsx'

    if not path.is_file():
        path = Path(__file__).parent.parent / 'metadata.xlsx'
    
    if not path.is_file():
        raise FileNotFoundError('metadata.xlsx file not found in corpus folder.')
    
    df = pd.read_excel(path)
    return df


def concatTables(metadataDF:pd.DataFrame) -> pd.DataFrame:
    root = Path('corpus') / 'tables'
    concatDF = None
    # st = pc()

    # for index, csv in tqdm(enumerate(metadataDF['linkTables']), ncols=80, desc="Loading csv's", total=len(metadataDF)):
    for index, csv in enumerate(metadataDF['linkTables']):
        tablePath = root / csv

        df = pd.read_csv(tablePath)
        df.insert(0, 'TEXT_ID', index)

        if concatDF is None:
            concatDF = df
        
        else:
            concatDF = pd.concat([concatDF, df], axis=0)

    # print(round(pc() - st,4), end='\t')
    return concatDF


def concatTables_async(metadataDF:pd.DataFrame) -> pd.DataFrame:
    # only slightly faster than single thread

    root = Path('corpus') / 'tables'
    concatDF = None
    st = pc()
    index = 0

    def func(csv:str) -> pd.DataFrame:
        nonlocal index
        df = pd.read_csv(root / csv)
        df.insert(0, 'TEXT_ID', index)
        index += 1
        return df

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(func, csv) for csv in metadataDF['linkTables']}

        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
            except Exception as e:
                print(e)

            if concatDF is None:
                concatDF = df
            
            else:
                concatDF = pd.concat([concatDF, df], axis=0)
    
    # print(round(pc() - st,4))
    return concatDF


def getSentences(df:pd.DataFrame) -> pd.DataFrame:

    def joinFunc(iterable:Iterable) -> str:
        return ' '.join([str(i) for i in iterable])

    grouped = df[['SENTENCE_ID', 'TOKEN']].groupby('SENTENCE_ID').agg({'TOKEN': joinFunc,
                                                                       'SENTIMENT_SENTENCE': 'unique'})
    return grouped






