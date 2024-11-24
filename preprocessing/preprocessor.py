import sys
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
from tqdm import tqdm



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

    dataPath = Path('data')
    metadataPath = dataPath / 'metadata.csv'
    textPath = dataPath / 'corpus'

    df = pd.read_csv(metadataPath)

    corpus = loadCorpusData(df, textPath)
    
    treeTEI = getTEILiteStructure()
    tree = convertToXML(corpus, treeTEI)

    with open(dataPath / 'corpus.xml', 'wb') as file:
        tree.write(file, encoding='utf-8', xml_declaration=True)

