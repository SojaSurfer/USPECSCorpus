from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors

from analysis import loadMetadata


qualitativeColors = plotly.colors.qualitative.Plotly


def concatTables(metadataDF:pd.DataFrame) -> pd.DataFrame:
    root = Path('corpus') / 'tables'
    concatDF = None

    for index, csv in tqdm(enumerate(metadataDF['linkTables']), ncols=80, desc="Loading csv's", total=len(metadataDF)):
    # for index, csv in enumerate(metadataDF['linkTables']):
        tablePath = root / csv

        df = pd.read_csv(tablePath)
        df.insert(0, 'TEXT_ID', index)

        if concatDF is None:
            concatDF = df
        
        else:
            concatDF = pd.concat([concatDF, df], axis=0)

    return concatDF


def sentimentBoxplotSpeaker(show: bool = True) -> None:

    metadataDF: pd.DataFrame = loadMetadata()
    fig = go.Figure()

    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]

        result = concatTables(df)

        result = result[['TEXT_ID', 'SENTIMENT_SENTENCE']].groupby('TEXT_ID').agg('mean')

        fig.add_trace(go.Box(
                y=result['SENTIMENT_SENTENCE'],
                name=speaker,
            ))
        

    fig.update_layout(
        title='Sentiment Analysis per Speaker',
        xaxis_title='Text ID',
        yaxis_title='Average Sentiment',
        legend_title='Speaker',
        #boxmode='group',  # Group the boxes together
        #boxgap=0.1,  # Decrease the gap between boxes
        boxgroupgap=0.1 
    )

    if show:
        fig.show()
    else:
        fig.write_image(Path('corpus') / 'visualizations' / 'sentiment_boxplot_speaker.png', height=1200, width=1800)
    return None


def sentimentBoxplotYear(show: bool = True) -> None:

    metadataDF: pd.DataFrame = loadMetadata()
    fig = go.Figure()

    years = range(2007, 2025, 1)

    for year in years:

        df = metadataDF[metadataDF['date'].dt.year == year]
        if df.empty:
            continue
        result = concatTables(df)

        result = result[['TEXT_ID', 'SENTIMENT_SENTENCE']].groupby('TEXT_ID').agg('mean')

        fig.add_trace(go.Box(
                y=result['SENTIMENT_SENTENCE'],
                name=year,
                marker=dict(color=qualitativeColors[0]), 
                showlegend=False
            ))
        

    fig.update_layout(
        title='Sentiment Analysis per Year',
        xaxis_title='Year',
        yaxis_title='Average Sentiment',
        #boxmode='group',  # Group the boxes together
        #boxgap=0.1,  # Decrease the gap between boxes
        boxgroupgap=0.1 
    )

    if show:
        fig.show()
    else:
        fig.write_image(Path('corpus') / 'visualizations' / 'sentiment_boxplot_years.png', height=1200, width=1800)
    return None



if __name__ == '__main__':

    sentimentBoxplotYear(show=False)