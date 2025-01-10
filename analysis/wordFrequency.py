import re
import string
import sys
from pathlib import Path
from typing import Generator
from collections import defaultdict, Counter
import json

import spacy
from nltk import FreqDist
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from analysis import loadMetadata, textGenerator, concatTables

periodLookup = {i+1: i * 4 + 2008 for i in range(5)}


def computePeriodMetrics(batchsize: int = 200) -> defaultdict[int, defaultdict[str, dict[str, float]]]:
    metadataDF: pd.DataFrame = loadMetadata()

    nlpSpacy = spacy.load("en_core_web_sm", enable=['tokenizer'])
    result = {}

    for period in metadataDF['period'].unique():
        periodDF = metadataDF[metadataDF['period'] == period]
        result[int(period)] = defaultdict(dict)

        for speaker in periodDF['speaker'].unique():
            df = periodDF[periodDF['speaker'] == speaker]

            generator = textGenerator(df, clearText=True)
            ttrList = []
            tokenCount = 0
            speechesCount = len(df)
            speakerTokens = []

            for text in generator:
                doc = nlpSpacy(text)
                tokens = [token.text.casefold() for token in doc if token.is_alpha]
                tokenCount += len(tokens)
                speakerTokens.extend(tokens)
                batchGenerator = tokenBatchGenerator(tokens, batchsize=batchsize)

                for batch in batchGenerator:
                    ttr = len(set(batch)) / len(batch)
                    ttrList.append(ttr)

            sttr = sum(ttrList) / len(ttrList) if ttrList else 0
            counter = Counter(speakerTokens)
            hapaxLegomena = sum(1 for count in counter.values() if count == 1)


            result[period][speaker]['speechesCount'] = speechesCount
            result[period][speaker]['STTR'] = sttr
            result[period][speaker]['tokenCount'] = tokenCount
            result[period][speaker]['hapaxLegomena'] = hapaxLegomena

    return result


def tokenBatchGenerator(doc, batchsize:int = 200) -> Generator[list[str], None, None]:

    spacyClass = False
    if hasattr(doc, 'text'):
        spacyClass = True

    batchAmount = len(doc) // batchsize

    for i in range(batchAmount):
        batchSlice = slice(i * batchsize, (i+1) * batchsize)

        if spacyClass:
            yield [tk.text for tk in doc[batchSlice]]
        else:
            yield [tk for tk in doc[batchSlice]]


def plotPeriodMetrics(metrics:dict, show: bool = False) -> None:

    electionYears = list(range(2008,2028,4))

    for period, pMetrics in metrics.items():
        fig = make_subplots(rows=4, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}],
                ],
            column_titles=list(pMetrics.keys()),
        )

        for i, (speaker, sMetrics) in enumerate(pMetrics.items(), 1):

            fig.add_trace(go.Indicator(
                mode='number',
                value=sMetrics['speechesCount'],
                title='Speeches',
                ), row=1,col=i)
            
            fig.add_trace(go.Indicator(
                mode='number',
                value=sMetrics['tokenCount'],
                title='Tokens',
                ), row=2,col=i)
            
            fig.add_trace(go.Indicator(
                mode='number',
                value=sMetrics['STTR'],
                title='STTR',
                ), row=3,col=i)
            
            fig.add_trace(go.Indicator(
                mode='number',
                value=sMetrics['hapaxLegomena'],
                title='Hapax Legomena',
                ), row=4,col=i)
            
        
        fig.update_layout(
            title={
                'text': f'{electionYears[int(period)-1]} Election Speeches Metrics',
                'font': {'size': 26},
                'x': 0.5,
                'xanchor': 'center',
            }
        )
        if show:
            fig.show()
        else:
            fig.write_image(Path(f'analysis/coreMetrics/metricsPeriod{period}.png'), height=1000, width=1000)

    return None


def plotWordFrequencySpeaker(amount: int = 10, show: bool = False) -> None:
    metadataDF: pd.DataFrame = loadMetadata()

    posTags = {'Nouns': ('NOUN', 'PROPN'), 'Adjectives': ('ADJ', 'ADV'), 'Verbs': ('VERB', 'AUX')}

    for period in tqdm(metadataDF['period'].unique(), ncols=80, desc='Plot word frequency'):
        periodDF = metadataDF[metadataDF['period'] == period]

        for speaker in periodDF['speaker'].unique():
            df = periodDF[periodDF['speaker'] == speaker]

            tokenTbl: pd.DataFrame = concatTables(df)
            
            fig = make_subplots(rows=1, cols=len(posTags), specs=[[{'type': 'xy'}] * len(posTags)],
                                column_titles=list(posTags.keys()))

            for i, tags in enumerate(posTags.values(), 1):
                posTbl = tokenTbl[tokenTbl['POS'].isin(tags)]
                fdist = FreqDist(posTbl['LEMMA'].str.lower())
                mostCommon = fdist.most_common(amount)
   

                fig.add_trace(go.Bar(
                    x=[w[0] for w in mostCommon],
                    y=[w[1] for w in mostCommon],
                    showlegend=False
                ), row=1, col=i)
            
            fig.update_layout(title={
                    'text': f'Word Frequency {speaker} \u2012 {periodLookup[period]} Election',
                    'x': 0.5,
                    'xanchor': 'center'
                })

            if show:
                fig.show()
            else:
                fig.write_image(Path(f'analysis/coreMetrics/Period{period}_{speaker}.png'), height=900, width=1600)

    return None



import gensim.similarities

gensim.similarities.Similarity()


if __name__ == '__main__':

    metrics = computePeriodMetrics()

    # with open('periodMetrics.json', 'r') as f:
    #     metrics = json.load(f)
    
    # plotPeriodMetrics(metrics)
    plotWordFrequencySpeaker()