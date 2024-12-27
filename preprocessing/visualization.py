import sys
from pathlib import Path
from datetime import date
import json

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.colors

from preprocessing.corpusStructure import ElectionPeriod



class Plotter():

    def __init__(self) -> None:

        self.discreteColors = plotly.colors.qualitative.D3


        self.root = Path('corpus')
        self.vizPath = self.root / 'visualizations'
        self.height = 1200
        self.width = 1800
        self.fontsize = 24

        self.metadataDF = pd.read_excel(self.root / 'metadata.xlsx')
        self.metadataDF['count'] = 1

        with open(Path('preprocessing') / 'resources.json', 'r') as f:
            data = json.load(f)
        
        self.stateAbbrev = data['state_abbrev']

        return None

    def plotAll(self, show: bool = False) -> None:
        
        self.plotSpeechesPerCandidate(show)
        self.plotSpeechesPerElection(show)
        self.plotChoropleth(show)
        self.plotCorpusStatistics(show)

        return None


    def plotSpeechesPerElection(self, show: bool = False) -> None:

        self.metadataDF['date'] = pd.to_datetime(self.metadataDF['date']) 
        # self.metadataDF['date'] = self.metadataDF['date'].dt.date
        self.metadataDF['count'] = 1

        speakerColorMap = {speaker: color for speaker, color in zip(self.metadataDF['speaker'].unique(), self.discreteColors)}


        df = self.metadataDF.groupby(['party', 'speaker', pd.Grouper(key='date', freq='ME')]).agg({'count': 'sum',
                                                                                            'period': 'median',
                                                                                            })
        df.reset_index(inplace=True)

        df['date'] = df['date'].dt.strftime('%Y-%m')


        generator = ElectionPeriod.generatePeriods(date(2007, 1, 1))

        fig = make_subplots(rows=5, cols=1,
            specs=[[{'type': 'scatter'}],
                [{'type': 'scatter'}],
                [{'type': 'scatter'}],
                [{'type': 'scatter'}],
                [{'type': 'scatter'}],],
            row_titles=[f'{electionPeriod.end.year} Election' for electionPeriod in generator],
        )


        for i in range(1, 6):
            
            period_df = df[df['period'] == i]
            area_fig = px.area(period_df, x='date', y='count', line_group='speaker', color='speaker',
                            color_discrete_map=speakerColorMap
                            )

            for trace in area_fig.data:
                trace.update(legendgroup=f'Period {i}', showlegend=True) 
                fig.add_trace(trace, row=i, col=1)


        fig.update_xaxes(range=['2007-01', '2008-11'], row=1, col=1)
        fig.update_xaxes(range=['2011-01', '2012-11'], row=2, col=1)
        fig.update_xaxes(range=['2015-01', '2016-11'], row=3, col=1)
        fig.update_xaxes(range=['2019-01', '2020-11'], row=4, col=1)
        fig.update_xaxes(range=['2023-01', '2024-11'], row=5, col=1)

        fig.update_yaxes(range=[0, 50])
        fig.update_layout(title= {'text': 'Speeches per Election Period', 
                                  'font': {'size': self.fontsize}},
                        legend_title=dict(
                            text="Speaker by Period",)
        )
        if show:
            fig.show()
        else:
            fig.write_image(self.vizPath / 'speeches_per_period.png', height=self.height, width=self.width)
        return None


    def plotSpeechesPerCandidate(self, show: bool = False) -> None:

        df = self.metadataDF[['speaker', 'party', 'count']].groupby(['party', 'speaker']).agg({'count': 'sum'})
        df.reset_index(inplace=True)


        fig = px.sunburst(df,
                        values='count',
                        path=['party', 'speaker'],
        )

        fig.update_traces(textinfo='label+value',
                        insidetextorientation='horizontal',
                        rotation=90,
        )
        
        fig.update_layout(title={'text': 'Speeches per Candidate',
                                 'font': {'size': self.fontsize}}
        )

        if show:
            fig.show()
        else:
            fig.write_image(self.vizPath / 'speeches_per_candidate.png', height=self.height, width=self.width)
        return None


    def plotChoropleth(self, show: bool = None) -> None:
        geo = self.metadataDF[['state', 'count']].groupby('state').agg({'count': 'sum'})
        geo.reset_index(inplace=True)
        geo['state_abbrev'] = geo['state'].map(self.stateAbbrev)
        geo = geo.dropna(subset=["state_abbrev"])
        geo.rename(columns={'count': 'Speeches'}, inplace=True)


        fig = px.choropleth(
            geo,
            locations='state_abbrev',
            locationmode='USA-states',
            color='Speeches',
            color_continuous_scale='Viridis',
            scope='usa',
            labels='Speeches'
        )

        fig.update_layout(title={'text': 'Speeches per State',
                                 'font': {'size': self.fontsize}}
        )

        if show:
            fig.show()
        else:
            fig.write_image(self.vizPath / 'choropleth.png', height=self.height, width=self.width)
        return None


    def plotCorpusStatistics(self, show: bool = False) -> None:

        fig = make_subplots(rows=2, cols=2,
            specs=[[{'rowspan': 2, 'type': 'scatter'}, {'type': 'indicator'}], 
                [None, {'type': 'indicator'}]],

            column_titles=['Histogram of Tokens per Speech', None],
            column_widths=[0.85, 0.15],
        )

        fig.add_trace(go.Histogram(
            x=self.metadataDF['tokens'],
            name='Tokens',
            nbinsx=40),
            row=1, col=1)

        fig.update_yaxes(range=[0,None], title='Frequency',
                        row=1, col=1)
        fig.update_xaxes(range=[0,None], title='Number of Tokens',
                        row=1, col=1)


        fig.add_trace(go.Indicator(
            mode='number',
            value=self.metadataDF['tokens'].sum(),
            title='Corpus Size [Tokens]',
            ),
            row=1,col=2)
        
        fig.add_trace(go.Indicator(
            mode='number',
            value=len(self.metadataDF['tokens']),
            title='Total Speeches',
            ),
            row=2,col=2)


        fig.update_layout(
            bargap=0.2,
            title='Corpus Statistics'
        )

        if show:
            fig.show()
        else:
            fig.write_image(self.vizPath / 'corpus_statistics.png', height=self.height, width=self.width)
        return None



if __name__ == '__main__':

    plotter = Plotter()

    plotter.plotAll()
