import string
import sys

import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from nltk import FreqDist
import pandas as pd
import seaborn as sns



# Parse the XML file
tree = ET.parse('data/corpus.xml')
root = tree.getroot()

namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}


df = pd.read_csv('data/metadata.csv')
speakers = df['speaker'].unique()

maxWords = 30
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
    # fdist.plot(maxWords, cumulative=False, title=)

    #ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
    ax.set_xlim([-0.5, maxWords-0.5])
    ax.grid(which='major', axis='y')
    ax.set_title(speaker)

plt.suptitle('Most frequent NN words')
plt.tight_layout()
plt.show()
plt.close()