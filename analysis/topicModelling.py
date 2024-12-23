import sys
from string import punctuation

import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS as stopwords
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
from tqdm import tqdm

from analysis import loadMetadata, concatTables



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


def topicModelling(numTopics:int, stopwords:set, iterations:int = 2_000) -> None:

    metadataDF: pd.DataFrame = loadMetadata()

    for speaker in metadataDF['speaker'].unique():

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords | punctuation)]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

        tokenTbl = tokenTbl[['TEXT_ID', 'ID', 'SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)

        documents = tokenTbl.groupby('TEXT_ID')['LEMMA'].apply(list).values

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
        pyLDAvis.save_html(vis, f'lda_{speaker}.html')
    
    return None



if __name__ == '__main__':

    punctuation = set(punctuation)
    punctuation |= {'--', '...', r'\u2013', r'\u2014', 'Dr.', '–', 'MR', 'Hi', 'Mr.', 'cheer', 'Cheer', 'applause', 'Applause', 'APPLAUSE', 'sir', 'Sir'}

    numTopics = 7  # maybe less
    # ideas: use only lowercase, remove candidate names

    topicModelling(numTopics, stopwords | punctuation)