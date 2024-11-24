import string

import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from nltk import FreqDist


# Parse the XML file
tree = ET.parse('data/corpus.xml')
root = tree.getroot()

namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}


# get most frequent words
# Find all <w> tags with the attribute pos='NN' within the namespace
words = root.findall(".//tei:w[@pos='NN']", namespaces)

maxWords = 30
punctuations = string.punctuation + '—'

fdist = FreqDist([word.text for word in words if not word.text in punctuations])

fig, ax = plt.subplots(figsize=(12, 8))
ax = fdist.plot(maxWords, cumulative=False, title='Most frequent NN words')

ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
ax.set_xlim([-0.5, maxWords-0.5])

plt.tight_layout()
plt.show()
plt.close()