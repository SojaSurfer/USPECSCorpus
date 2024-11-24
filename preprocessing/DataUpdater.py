from pathlib import Path
import os
from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessor import *



class DataUpdater():

    def __init__(self):
        
        self.paths = self._getPaths()
        self.checkDataStructure()

        self.df = self._getUpdatedDF()

        return None


    def checkDataStructure(self) -> None:

        if not self.paths['data'].exists():
            raise FileNotFoundError(f'Data directory does not exist. Add the data folder to the project root.')
        
        if not self.paths['csv'].exists() or not self.paths['excel'].exists():
            raise FileNotFoundError(f'No metadata file found. Add metadata.csv or metadata.xlsx to the data folder.')

        if not self.paths['corpus'].exists():
            raise FileNotFoundError(f'No corpus folder found. Add a corpus folder to the data folder.')

        self.initialCorpusSize = len(list(self.paths['corpus'].rglob('*')))
        
        return None


    def _getUpdatedDF(self) -> pd.DataFrame:
        modtimeCSV = os.path.getmtime(self.paths['csv'])
        modtimeExcel = os.path.getmtime(self.paths['excel'])

        if modtimeExcel > modtimeCSV:
            df = pd.read_excel(self.paths['excel'])
        else:
            df = pd.read_csv(self.paths['csv'])
       
        return df

    @staticmethod
    def _getPaths() -> dict[str, Path]:
        dataPath = Path(__file__).parent.parent / 'data'

        paths = {'data': dataPath,
                'csv': dataPath / 'metadata.csv',
                'excel': dataPath / 'metadata.xlsx',
                'corpus': dataPath / 'corpus'}

        return paths


    def update(self) -> None:
        self.updateCorpus()
        self.updateMetadata()
        self.updateVisualization()
        return None


    def updateXML(self) -> None:
        
        df = pd.read_csv(self.paths['csv'])
        corpus = loadCorpusData(df, self.paths['corpus'])

        treeTEI = getTEILiteStructure()
        tree = convertToXML(corpus, treeTEI)

        with open(self.paths['data'] / 'corpus.xml', 'wb') as file:
            tree.write(file, encoding='utf-8', xml_declaration=True)

        return None


    def updateMetadata(self) -> None:
        self.df.to_csv(self.paths['csv'], index=False)
        self.df.to_excel(self.paths['excel'], index=False)
        return None


    def updateVisualization(self, debug:bool = False) -> None:

        output = self.paths['data'] / 'visualization.png' 

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
        
        for file in self.paths['corpus'].rglob('*.txt'):
            if not int(file.stem) in self.df.iloc[:, 0].values:
                os.remove(file)

        self.updatedCorpusSize = len(list(self.paths['corpus'].rglob('*')))
        print(f'Removed {self.initialCorpusSize - self.updatedCorpusSize} files from the corpus.')
        return None




if __name__ == '__main__':
    
    updater = DataUpdater()
    
    updater.update()
    updater.updateXML()
