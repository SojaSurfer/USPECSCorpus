from pathlib import Path
import os
from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class DataUpdater():

    def __init__(self):
        
        self.paths = self._getPaths()
        
        self.df = pd.read_csv(self.paths['csv'])

        return None


    @staticmethod
    def _getPaths() -> dict[str, Path]:
        dataPath = Path().joinpath('.', 'data')

        paths = {'path': dataPath,
                'csv': dataPath / 'metadata.csv',
                'excel': dataPath / 'metadata.xlsx',
                'corpus': dataPath / 'corpus'}

        return paths


    def update(self) -> None:
        self.updateVisualization()
        self.updateCorpus()
        return None


    def updateVisualization(self, debug:bool = False) -> None:

        output = self.dataPath / 'visualization.png' 

        partyColor = {'John McCain': 'red', 'Barack Obama': 'blue', 
                        'Mitt Romney': 'red', 'Hillary Clinton': 'blue', 
                        'Donald J. Trump (1st Term)': 'red', 'Joseph R. Biden, Jr.': 'blue', 
                        'Kamala Harris': 'blue'}


        sns.set_style('darkgrid')
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # speeches per candidate
        speakers = self.df['speaker'].value_counts()

        ig = itemgetter(*speakers.index)

        axes[0].bar(speakers.index, speakers.values, color=ig(partyColor))
        axes[0].set_title('Number of Speeches per Candidate')


        # speeches per date
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')
        self.df['month'] = self.df['date'].dt.to_period('M')  # Convert to monthly period

        speechesDate = self.df['month'].value_counts(sort=False)

        axes[1].scatter(speechesDate.index.to_timestamp(), speechesDate.values, color='orange')

        axes[1].set_title('Speeches in the Dataset')
        axes[1].set_ylim(bottom=0)
        axes[1].set_xlim([speechesDate.index[0], speechesDate.index[-1]])


        # speeches per state
        states = self.df['state'].value_counts()

        axes[2].bar(states.index, states.values, color='green')
        axes[2].set_title('State of the Speech')
        axes[2].set_xlim([0, len(states)])
        axes[2].tick_params(axis='x', rotation=45, labelsize=8) 
        
        # axes[2].pie(states.values, labels=states.index)
        # axes[2].set_title('State of the Speech')


        plt.tight_layout()
        if debug:
            plt.show()
        else:
            plt.savefig(output)
        plt.close()
        return None


    def updateCorpus(self):
        
        for val in self.df.iloc[:, 0]:

            deletePath = self.paths['corpus'] / f'{val:04d}.txt'
            # os.remove(deletePath)
        
        return None




if __name__ == '__main__':

    
    updater = DataUpdater()
    updater.update()
