# needs to be modified to fit new csv format !!

import re
import string
import sys
from pathlib import Path

import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from nltk import FreqDist, collocations
import pandas as pd
import seaborn as sns
from tqdm import tqdm




def createNetworkCsv(maxWords:int, root, speakers:list, postag:str = 'NN') -> None:

    punctuations = string.punctuation + '—'
    df = pd.DataFrame(columns=['Source', 'Type', 'Target', 'Weight'])

    for speaker in speakers:

        words = root.findall(f".//tei:div[@speaker='{speaker}']//tei:w[@pos='{postag}']", namespaces)

        fdist = FreqDist([word.text for word in words if not word.text in punctuations])
        mostCommon = fdist.most_common(maxWords)

        presidentRow = {'Source': speaker}

        for (word, count) in mostCommon:
            row = presidentRow | {'Type': 'Directed', 'Target': word, 'Weight': count}

            df.loc[len(df)] = row


    df.to_csv(f'presidentsTop{maxWords}{postag}words.csv', index=False)

    return None


def plotFrequencyBySpeaker(maxWords:int, root, speakers:list) -> None:
    punctuations = string.punctuation + '—'

    sns.set_style('darkgrid')
    fig, axes = plt.subplots(4, 2, figsize=(17, 11))

    for i, ax in enumerate(axes.flatten()):

        if i >= len(speakers):
            break

        speaker = speakers[i]

        # get most frequent words
        # Find all <w> tags with the attribute pos='NN' within the namespace
        words = root.findall(f".//tei:div[@speaker='{speaker}']//tei:w[@pos='NN']", namespaces)

        fdist = FreqDist([word.text for word in words if not word.text in punctuations])
        mostCommon = fdist.most_common(maxWords)
        words = [m[0] for m in mostCommon]
        count = [m[1] for m in mostCommon]


        ax.plot(words, count)

        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
        ax.set_xlim([-0.5, maxWords-0.5])
        ax.grid(which='major', axis='y')
        ax.set_title(speaker)


    plt.suptitle('Most frequent NN words')
    plt.tight_layout()
    plt.show()
    plt.close()

    return None


def plotFrequencyByYear(maxWords:int, root, years:list) -> None:
    punctuations = string.punctuation + '—'
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(6, 2, figsize=(17, 11))


    for i, ax in enumerate(axes.flatten()):

        if i >= len(years):
            break

        year = years[i]

        # get most frequent words
        # Find all <w> tags with the attribute pos='NN' within the namespace
        query = f".//tei:div[@year='{year}']//tei:w[@pos='NN']"

        words = root.findall(query, namespaces)


        fdist = FreqDist([word.text for word in words if not word.text in punctuations])
        mostCommon = fdist.most_common(maxWords)



        words = [m[0] for m in mostCommon]
        count = [m[1] for m in mostCommon]


        ax.plot(words, count)

        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
        ax.set_xlim([-0.5, maxWords-0.5])
        ax.grid(which='major', axis='y')
        ax.set_title(year)


    plt.suptitle('Most frequent NN words by year')
    plt.tight_layout()
    plt.show()
    plt.close()
    return None


def plotCollocations(root, text:str, pattern:str) -> None:

    def printMatchWithContext(text, pattern, context=20):
        matches = re.finditer(pattern, text)

        for match in matches:
            start, end = match.start(), match.end()
            # Extract context
            before = text[max(0, start - context):start].replace('\n', ' ')  # Up to `context` characters before
            after = text[end:end + context].replace('\n', ' ')              # Up to `context` characters after
            # Print the match with context
            print(f"...{before}{match.group()}{after}...")
    
        return None

    # printMatchWithContext(text, pattern)
    speaker = 'Donald J. Trump (1st Term)'
    words = root.findall(f".//tei:div[@speaker='{speaker}']//", namespaces)

    bigramMeasures = collocations.BigramAssocMeasures()

    finder = collocations.BigramCollocationFinder.from_words([w.text for w in words], window_size=20)

    result = finder.nbest(bigramMeasures.pmi, 10)
    print(result)

    return None


def wordBefore(root):
    import nltk

    stopwords = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['this', "'s", "applause"]

    speaker = 'Donald J. Trump (1st Term)'
    words = root.findall(f".//tei:div[@speaker='{speaker}']//", namespaces)

    tokens = [w.text for w in words if not w.text.lower() in stopwords and (w.get('pos').startswith('J') or w.get('pos').startswith('N'))]

    target_word = 'guy'
    preceding_words = []

    for i in range(1, len(tokens)):
        if tokens[i].lower() == target_word:
            preceding_words.append(tokens[i - 1].lower())  # Add the word before the target word

    # Count the most common preceding words
    freq_dist = FreqDist(preceding_words)

    # Display the most common words before the target word
    most_common = freq_dist.most_common(20)  # Adjust the number as needed
    print("Most common words before '{}':".format(target_word))
    for word, count in most_common:
        print(f"{word}: {count}")

    plt.pie([x[1] for x in most_common], labels=[x[0] for x in most_common])
    plt.title('Most used adjectives after "guy"')
    plt.show()
    plt.close()


    tokens = [w.text for w in words if not w.text.lower() in stopwords and (w.get('pos').startswith('J'))]

    # Count the most common preceding words
    freq_dist = FreqDist(tokens)

    # Display the most common words before the target word
    most_common = freq_dist.most_common(10)  # Adjust the number as needed

    plt.pie([x[1] for x in most_common], labels=[x[0] for x in most_common])
    plt.title('Most used adjectives')
    plt.show()
    plt.close()

    return None


if __name__ == '__main__':

    # Parse the XML file
    dataPath = Path('data')


    tree = ET.parse(dataPath / 'corpus.xml')
    root = tree.getroot()

    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}


    df = pd.read_csv(dataPath / 'metadata.csv')
    speakers = df['speaker'].unique()

    maxWords = 30
    years = [2007, 2008, 
             2011, 2012,
             2015, 2016,
             2019,2020,2021,2022,
             2023,2024]


    # createNetworkCsv(maxWords, root, speakers)


    # trumpDF = df[df['speaker'] == 'Donald J. Trump (1st Term)']
    
    # pattern = '[Gg]uy'
    # for id_ in trumpDF['ID'].values:
    
    #     with open(f'data/corpus/{id_:04d}.txt', 'r') as f:
    #         text = f.read()
    
    # plotCollocations(root, None,None)
    wordBefore(root)