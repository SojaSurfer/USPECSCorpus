import sys
from string import punctuation
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import pandas as pd
import numpy as np
import gensim
from nltk import sent_tokenize
from gensim.parsing.preprocessing import STOPWORDS as stopwords
from gensim import models
import pyLDAvis.gensim_models
from tqdm import tqdm
from rich import traceback
traceback.install()

from analysis import loadMetadata, concatTables, getSentences, concatTexts



# analyze topics - main topics for each document in the collection
def analyse_topics(ldamodel, corpus, text) -> tuple[dict, ...]:
    main_topic = {}
    percentage = {}
    keywords = {}
    text_snippets = {}
    
    for i, topic_list in enumerate(ldamodel [corpus]):
        topic = topic_list[0]
        topic = sorted(topic, key = lambda x: (x[1]), reverse = True)
        
        for j, (topic_num, prop_topic) in enumerate(topic):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp[:5]])
                main_topic[i] = int(topic_num)
                percentage[i] = round(prop_topic, 4)
                keywords[i] = topic_keywords
                text_snippets[i] = text[i][:8]
            else:
                break
    return main_topic, percentage, keywords, text_snippets


def topicModelling(numTopics:int, stopwords:set, iterations:int = 2_000, withNGrams: bool = False,
                   show: bool = False) -> None:

    metadataDF: pd.DataFrame = loadMetadata()

    for speaker in metadataDF['speaker'].unique():

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
        tokenTbl = tokenTbl[~tokenTbl['LEMMA'].isin(stopwords)]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

        tokenTbl = tokenTbl[['TEXT_ID', 'ID', 'SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)

        documents = tokenTbl.groupby('TEXT_ID')['LEMMA'].apply(list).values

        # tokens to bigrams and trigrams
        if withNGrams:
            bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
            trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))
            documents = bigram[documents]
            documents = trigram[documents]


        ## topic modelling
        dictionary = gensim.corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n = 10000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(documents, ncols=80)]


        # run LDA
        lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                                    id2word=dictionary,
                                                    num_topics=numTopics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=1000,
                                                    passes=10,
                                                    alpha='symmetric',
                                                    iterations=iterations,
                                                    per_word_topics=True)
        

        # analyze topics - print out main topic for each document in the collection
        # main_topic, percentage, keywords, text_snippets = analyse_topics(lda_model, bow_corpus, documents)

        # rows = [['ID', 'Main Topic', 'Contribution (%)', 'Keywords', 'Snippet' ]]

        # for idx in range(numTopics):
        #     rows.append([str(idx), f"{main_topic.get(idx)}",
        #                 f"{percentage.get(idx):.4f}",
        #                 f"{keywords.get(idx)}\n",
        #                 f"{text_snippets.get(idx)}"])
        # columns = zip(*rows)
        # column_width = [max(len(item) for item in col) for col in columns]
        # for row in rows:
        #     print(row)


        vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary=lda_model.id2word)
        if show:
            pyLDAvis.show(vis)
            return None
        else:
            pyLDAvis.save_html(vis, f'lda_{speaker}.html')
    
    return None


def ensembleTopicModelling(numTopics:int, stopwords:set, iterations:int = 2_000) -> None:

    metadataDF: pd.DataFrame = loadMetadata()

    for speaker in metadataDF['speaker'].unique():

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords | set(punctuation))]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

        tokenTbl = tokenTbl[['TEXT_ID', 'ID', 'SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)

        documents = tokenTbl.groupby('TEXT_ID')['LEMMA'].apply(list).values

        ## topic modelling
        dictionary = gensim.corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n = 10000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(documents, ncols=80)]


        ensemble = models.EnsembleLda(corpus=bow_corpus,
                                            id2word=dictionary,
                                            num_topics=numTopics,
                                            random_state=100,
                                            topic_model_class=models.LdaModel,
                                            ensemble_workers=3,
                                            distance_workers=3,
        )
        print_topics(ensemble.asymmetric_distance_matrix, dictionary)

        return None
    

def print_topics(phi, dictionary, top_n=10):
    """
    Print the top words for each topic in the EnsembleLDA model.

    Args:
        phi (numpy.ndarray): Topic-word distribution matrix.
        dictionary (gensim.corpora.Dictionary): Gensim dictionary mapping word IDs to tokens.
        top_n (int): Number of top words to display for each topic.
    """
    for topic_idx, topic_dist in enumerate(phi):
        # Get top word indices for the topic
        top_word_ids = np.argsort(-topic_dist)[:top_n]  # Sort by descending probability
        top_words = [dictionary[word_id] for word_id in top_word_ids]  # Map IDs to words
        top_probs = topic_dist[top_word_ids]  # Get probabilities for the top words

        # Print the topic
        print(f"Topic {topic_idx + 1}:")
        for word, prob in zip(top_words, top_probs):
            print(f"  {word}: {prob:.4f}")
        print()  # Blank line between topics


def authorTopicModelling(numTopics:int) -> None:
    path = Path('corpus/tables')
    # gensim.models.AuthorTopicModel()

    metadataDF = loadMetadata()
    author2doc = defaultdict(list)
    texts = []

    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]

        for file in df['linkTables']:
            tablePath = path / file
            table = pd.read_csv(tablePath)
            tokens = filterTokens(table)
            texts.append(tokens)

            author2doc[speaker].append(len(texts) - 1)


    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]


    model = gensim.models.AuthorTopicModel(
        corpus=corpus,
        num_topics=numTopics,
        id2word=dictionary,
        author2doc=author2doc,
        update_every=1,
        chunksize=1000,
        iterations=1000,
        passes=10  # Number of training passes
    )

    # Step 4: Analyze Topics per Speaker
    # Get the topics associated with each speaker
    for author in model.author2doc.keys():
        print(f"Topics for {author}:")
        print(model.get_author_topics(author))
        print()

    # Step 5: Examine a specific topic
    # Get the top words for topic 0
    print("Top words for topic 0:")
    print(model.print_topics())


    return None


def filterTokens(tokenTbl:pd.DataFrame) -> list[str]:
    from string import punctuation
    punctuation = set(punctuation)
    punctuation |= {'--', '...', r'\u2013', r'\u2014', 'dr.', '–', 'mr', 'hi', 'mr.', 'cheer', 'applause', 'sir', 'cheers'}

    tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords | punctuation)]
    tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

    return tokenTbl['LEMMA'].tolist()


def trainBigrams():

    def joinFunc(iterable:Iterable) -> str:
        
        return list(iterable)

    metadataDF: pd.DataFrame = loadMetadata()
    sentences = []

    for speaker in metadataDF['speaker'].unique():

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords | set(punctuation))]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

        tokenTbl = tokenTbl[['TEXT_ID', 'ID', 'SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)
        tokenTbl.sort_values(by=['TEXT_ID', 'ID'])

        tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
        documents = tokenTbl[['TEXT_ID', 'SENTENCE_ID', 'LEMMA']].groupby(['TEXT_ID', 'SENTENCE_ID']).agg({'LEMMA': joinFunc})

        sentences.extend(documents['LEMMA'].values)


    bigramPhrases = models.Phrases(sentences,
                                   min_count=9,
                                   threshold=80,)
    trigramPhrases = models.Phrases(bigramPhrases[sentences],
                                   min_count=12,
                                   threshold=90,)
    
    bigram = models.phrases.Phraser(bigramPhrases)
    bigram.save(str(Path('analysis/topicModelling/bigram.model')))
    
    trigram = models.phrases.Phraser(trigramPhrases)
    trigram.save(str(Path('analysis/topicModelling/trigram.model')))


    # Analyse result
    dataBigrams = [bigram[doc] for doc in sentences]
    dataTrigrams = [trigram[doc] for doc in dataBigrams]

    foundBigrams = [word for sentence in dataBigrams for word in sentence if '_' in word]
    for bi in foundBigrams[:20]:
        print(bi)
    print()

    import re
    pattern = re.compile(r'.+_.+_.+')
    foundTrigrams = [word for sentence in dataTrigrams for word in sentence if '_' in word]
    for tri in set(foundTrigrams):
        if pattern.match(tri):
            print(tri)

    return None


if __name__ == '__main__':

    from string import punctuation
    punctuation = set(punctuation)
    punctuation |= {'--', '...', r'\u2013', r'\u2014', 'dr.', '–', 'mr', 'hi', 'mr.', 'cheer', 'applause', 'sir'}
    stopwords = stopwords | punctuation


    numTopics = 7  # maybe less
    # ideas: use only lowercase, remove candidate names

    # trainBigrams()
    topicModelling(numTopics, stopwords, withNGrams=True)
    # authorTopicModelling(numTopics)
    # ensembleTopicModelling(numTopics, stopwords | set(punctuation))