from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

from analysis.analysis import loadMetadata, concatTables


qualitativeColors = plotly.colors.qualitative.Plotly

candidateLookup = {
    1: ('Barack Obama', 'John McCain'),
    2: ('Barack Obama', 'Mitt Romney'),
    3: ('Hillary Clinton', 'Donald J. Trump (1st Term)'),
    4: ('Joseph R. Biden, Jr.', 'Donald J. Trump (1st Term)'),
    5: ('Kamala Harris', 'Donald J. Trump (1st Term)'),
}



def sentimentBoxplotSpeaker(show: bool = True) -> None:

    metadataDF: pd.DataFrame = loadMetadata()
    fig = go.Figure()

    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]

        result = concatTables(df)
        # result = concatTables_async(df)

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


def sentimentViolinPeriod(show: bool = True) -> None:
    metadataDF: pd.DataFrame = loadMetadata()


    layout = {'side': ['negative', 'positive'],
              'color': ['blue', 'red']}
    yaxisRange = (-0.5, 1)
    periodList: list = metadataDF['period'].unique().astype(int).tolist()
    periodList.sort()
    periodList.remove(0)
    

    fig = make_subplots(rows=len(periodList), cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]] * len(periodList),
        column_titles=['Candidate Comprehension', 'Sentiment Process (speech avg per month)'],
        row_titles=[f'{year} Election' for year in range(2008, 2028, 4)],  # hardcoded periods...
        column_widths=[0.15, 0.85],
    )

    for period in tqdm(periodList, ncols=80, desc='Plotting'):
        df = metadataDF[metadataDF['period'] == period]

        for i, candidate in enumerate(candidateLookup[period]):
            candidateDF:pd.DataFrame = df[df['speaker'] == candidate]

            result = concatTables(candidateDF)

            # Violin plot
            violin = result[['TEXT_ID', 'SENTIMENT_SENTENCE']].groupby('TEXT_ID').agg('mean')
            violin.rename(columns={'SENTIMENT_SENTENCE': 'SENTIMENT_SPEECH'}, inplace=True)


            fig.add_trace(go.Violin(
                x=violin['SENTIMENT_SPEECH'] * 0,
                y=violin['SENTIMENT_SPEECH'],
                legendgroup=f'{candidateDF.iloc[-1]["date"].year} Election',
                width=0.6,
                name=candidate,
                side=layout['side'][i], line_color=layout['color'][i],
            ),row=period, col=1)

            fig.update_layout(
                legend_title=f'{candidateDF.iloc[-1]["date"].year} Election',
                legend=dict(groupclick="toggleitem")
            )

            # line plot
            result['date'] = result['TEXT_ID'].apply(lambda x: candidateDF.iloc[x]['date'])
            result['date'] = pd.to_datetime(result['date']) 

            result =  result[['date', 'TEXT_ID', 'SENTIMENT_SENTENCE']].groupby(['TEXT_ID', 'date']).agg(
                 {'SENTIMENT_SENTENCE': 'mean'})
            result.reset_index(inplace=True)

            result['SENTIMENT_SPEECH_STD'] = result['SENTIMENT_SENTENCE']
            scatter = result[['date', 'TEXT_ID', 'SENTIMENT_SENTENCE', 'SENTIMENT_SPEECH_STD']].groupby(
                            [pd.Grouper(key='date', freq='ME')]).agg(
                                {'SENTIMENT_SENTENCE': 'mean',
                                'SENTIMENT_SPEECH_STD': 'std',
                                'TEXT_ID': 'nunique'})
            
            scatter['SENTIMENT_SPEECH_STD'] = scatter['SENTIMENT_SPEECH_STD'].fillna(0.0)
            scatter.reset_index(inplace=True)
            scatter.rename(columns={'TEXT_ID': 'TEXT_COUNT', 'SENTIMENT_SENTENCE': 'SENTIMENT_SPEECH'}, inplace=True)
            scatter['date'] = scatter['date'].dt.strftime('%Y-%m')

            fig.add_trace(go.Scatter(
                x=scatter['date'],
                y=scatter['SENTIMENT_SPEECH'],
                name=candidate,
                showlegend=False,
                line=dict(color=layout['color'][i]) 
            ),row=period, col=2)


        fig.update_xaxes(title_text=f'{candidateDF.iloc[-1]["date"].year} Election', showticklabels=False, 
                         row=period, col=1)
        fig.update_yaxes(title_text='Average Sentiment', 
                         range=yaxisRange,
                         row=period, col=1)
        fig.update_yaxes(range=yaxisRange,
                         row=period, col=2)


    fig.update_layout(
        title='Sentiment Analysis per Election',
        legend_title='Candidates')

    if show:
        fig.show()
    else:
        fig.write_image(Path('corpus') / 'visualizations' / 'sentiment_period.png', height=1200, width=1800)
    return None




if __name__ == '__main__':

    # sentimentBoxplotSpeaker()
    # sentimentBoxplotYear()
    # sentimentViolinPeriod(show=True)