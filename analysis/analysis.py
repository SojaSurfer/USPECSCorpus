from typing import Iterable
from pathlib import Path

import pandas as pd



def loadMetadata() -> pd.DataFrame:
    path = Path('corpus') / 'metadata.xlsx'

    if not path.is_file():
        path = Path(__file__).parent.parent / 'metadata.xlsx'
    
    if not path.is_file():
        raise FileNotFoundError('metadata.xlsx file not found in corpus folder.')
    
    df = pd.read_excel(path)
    return df



def getSentences(df:pd.DataFrame) -> pd.DataFrame:

    def joinFunc(iterable:Iterable) -> str:
        return ' '.join([str(i) for i in iterable])

    grouped = df[['SENTENCE_ID', 'TOKEN']].groupby('SENTENCE_ID').agg({'TOKEN': joinFunc,
                                                                       'SENTIMENT_SENTENCE': 'unique'})
    return grouped






