from datetime import date, timedelta
from typing import Generator, Any
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
import numpy as np
from tqdm import tqdm
import spacy



class ElectionPeriod():
    """
    A class to represent an US election period.

    Attributes:
        period (int): The period number.
        start (date): The start date of the election period.
        end (date): The end date of the election period.

    Methods:
        generatePeriods(start: date = None) -> Generator['ElectionPeriod', Any, None]:
            Class method to generate election periods starting from a specified date.
    """

    def __init__(self, period:int, start:date, end:date) -> None:
        self.period = period
        self.start = start
        self.end = end

        return None
    
    def __str__(self):
        return f'{self.__class__.__qualname__}(period={self.period}, start={self.start}, end={self.end})'

    def __repr__(self):
        return f'{self.__class__.__qualname__}(period={self.period!r}, start={self.start!r}, end={self.end!r})'

    def __contains__(self, other:Any) -> bool:
        if isinstance(other, (date, np.datetime64)):
            if isinstance(other, np.datetime64):
                other = other.astype('M8[D]').astype(date)
            return self.start <= other <= self.end
        else:
            return NotImplementedError('Only implemented for date objects')

    @classmethod
    def generatePeriods(cls, start:date = None) -> Generator['ElectionPeriod', Any, None]:
        """Generate US election periods starting from a specified date. 
        Each election period begins in January of the year before the election and ends at the end of November of the election year.
        
        Args:
            start (date): The start date for the first election period. Defaults to January 1, 2007.

        Yields:
            ElectionPeriod: A named tuple containing the period number, start date, and end date of the election period.
        """
            
        electionPeriod = timedelta(days=699)
        nonElectionPeriod = timedelta(days=365*4 + 1) - electionPeriod

        start = start or date(2007, 1, 1)
        period = 1

        while start < date.today():
            end = start + electionPeriod
            yield cls(period, start, end)

            start = end + nonElectionPeriod
            period += 1

        return None


def addElectionPeriodToDF(metadataDF:pd.DataFrame) -> pd.DataFrame:

    gen = ElectionPeriod.generatePeriods(date(2007, 1, 1))
    electionPeriods = [next(gen) for _ in range(5)]

    def mapPeriods(speechDate:date) -> int|str:
        for elecPeriod in electionPeriods:
            if speechDate.date() in elecPeriod:
                return elecPeriod.period
        
        return ''


    metadataDF['date'] = pd.to_datetime(metadataDF['date'], format='%y-%M-%d') 
    metadataDF['period'] = metadataDF['date'].apply(mapPeriods)

    metadataDF['date'] = metadataDF['date'].dt.date
    metadataDF = metadataDF[['ID', 'speaker', 'party', 'date', 'period', 'state', 'city', 'population', 'title', 'citation', 'categories', 'linkText', 'linkTables', 'linkWeb']]

    return metadataDF


def createTables(metadataPath:str) -> None:
    """The function creates tables from the text files and saves them as csv files.
    The tables contain the following columns: SENTENCE_ID, TOKEN_ID, TOKEN, LEMMA, POS, MORPH, HEAD, DEPREL, NAMED_ENTITY, SENTIMENT_SENTENCE.
    """

    root = Path('corpus')
    root = Path('PresidencyScraperResult2025-01-17_Trump2')
    metadataDF = pd.read_excel(metadataPath)

    nlpSpacy = spacy.load("en_core_web_sm")
    sia = SentimentIntensityAnalyzer()

    cols = ['SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA', 'POS', 'MORPH', 'HEAD', 'DEPREL', 'NAMED_ENTITY', 'SENTIMENT_SENTENCE']

    for index, row in tqdm(metadataDF.iterrows(), total=len(metadataDF), ncols=80, desc='Tokenizing'):
        df = pd.DataFrame(columns=cols)
        df.index.name = 'ID'

        with open(root / 'texts' / row['linkTexts'], 'r') as file:
            text = file.read()
        

        sentences = sent_tokenize(text, language='english')
        

        for sentenceID, sentence in enumerate(sentences, 1):
            cleanSentence = sentence.replace('\n', '').replace('\r', '').replace('\t', '')

            doc = nlpSpacy(cleanSentence)
            sentiment = sia.polarity_scores(cleanSentence)['compound']

            for wordID, tk, in enumerate(doc, 1):
                ne = tk.ent_type_ or 'O'
                tableRow = (sentenceID, wordID, tk.text, tk.lemma_, tk.pos_, tk.morph, tk.head.i, tk.dep_, ne, sentiment)
                df.loc[len(df)] = tableRow

        
        df.to_csv(root / 'tables' / row['linkTables'])

    return None



## old xml structure
def loadCorpusData(metadataDF:pd.DataFrame, textPath:Path) -> list[dict]:
    """The function loads in the metadata and text files, tokenizes it and returns it."""

    corpus = []

    for index, row in tqdm(metadataDF.iterrows(), total=len(metadataDF), ncols=80, desc='Tokenizing'):
        filename = f'{row["ID"]:04d}.txt'

        with open(textPath / filename, 'r') as file:
            text = file.read()
        
        tokens = word_tokenize(text, language='english')
        posTags = pos_tag(tokens)

        corpus.append({'filename': filename,
                        'speaker': row['speaker'],
                        'title': row['title'],
                        'date': row['date'],
                        'year': row['date'].split('-')[0],
                        'state': row['state'],
                        'city': row['city'],
                        'population': str(row['population']),
                        'tokens': {'text': tokens,
                                   'pos': posTags
                                    }
                        })

    return corpus


def getTEILiteStructure() -> ET:
    """The function creates and returns the basic TEI lite structure as ElementTree."""

    root = ET.Element('TEI', {'xmlns': 'http://www.tei-c.org/ns/1.0'})

    # TEI header elems
    teiHeader = ET.SubElement(root, 'teiHeader')
    fileDesc = ET.SubElement(teiHeader, 'fileDesc')
    titleStmt = ET.SubElement(fileDesc, 'titleStmt')
    title = ET.SubElement(titleStmt, 'title')
    title.text = 'US Presidental Candidate Speeches'
    author = ET.SubElement(titleStmt, 'author')
    author.text = 'https://www.presidency.ucsb.edu/'

    publicationStmt = ET.SubElement(fileDesc, 'publicationStmt')
    publisher = ET.SubElement(publicationStmt, 'publisher')
    publisher.text = 'https://www.presidency.ucsb.edu/'
    date = ET.SubElement(publicationStmt, 'date')
    date.text = '2024-11-08'

    sourceDesc = ET.SubElement(fileDesc, 'sourceDesc')
    ET.SubElement(sourceDesc, 'bibl')

    # TEI text elems
    text = ET.SubElement(root, 'text')
    ET.SubElement(text, 'body')

    tree = ET.ElementTree(root)
    return tree
    

def convertToXML(corpus:list[dict], treeTEI:ET) -> ET:
    root = treeTEI.getroot()

    body = root.find('text').find('body')

    counter = 0

    for i, item in tqdm(enumerate(corpus, 1), total=len(corpus), ncols=80, desc='creating XML'):
        
        tokenList = item.pop('tokens')

        attributes = {'type': 'text', 'id': f't{i}'} | item
        divElem = ET.SubElement(body, 'div', attributes)

        for posTag in tokenList['pos']:
            counter += 1
            attrib = {'id': f'w{counter}', 'pos': posTag[1]}
            tokenElem = ET.SubElement(divElem, 'w', attrib)
            tokenElem.text = posTag[0]
    

    tree = ET.ElementTree(root)
    return tree
    



if __name__ == '__main__':
    
    # metadataPath = Path('corpus', 'metadata.xlsx')
    metadataPath = Path('PresidencyScraperResult2025-01-17_Trump2/metadata.xlsx')
    
    # create all csv tables for the text files listed in the metadata excel
    createTables(metadataPath)