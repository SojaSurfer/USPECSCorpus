import re
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterable
import pickle
import math

import pandas as pd
import plotly.express as px
import numpy as np
import gensim
from nltk import sent_tokenize
from gensim import models
import pyLDAvis.gensim_models
from tqdm import tqdm
from rich import traceback
traceback.install()

from analysis import loadMetadata, concatTables, getSentences, concatTexts


COLOR_MAP = {
    'Barack Obama': 'blue',
    'John McCain': 'red',
    'Mitt Romney': 'red',
    'Hillary Clinton': 'blue',
    'Donald J. Trump (1st Term)': 'red',
    'Joseph R. Biden, Jr.': 'blue',
    'Kamala Harris': 'blue'
}


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


def topicModelling(numTopics:int, stopwords:set, iterations:int = 2_000, withNGrams: bool = False) -> None:

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
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n = 10_000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(documents, ncols=80)]


        # run LDA
        st = time.perf_counter()
        lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus,
                                                    id2word=dictionary,
                                                    num_topics=numTopics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=1_000,
                                                    passes=10,
                                                    alpha='symmetric',
                                                    iterations=iterations,
                                                    per_word_topics=True)

        print(f'duration: {time.perf_counter() - st:.4f}')
        

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

        # wordDF = getTopicWords(lda_model)
        # wordDF.to_csv(f'analysis/topicModelling/topics/topics{speaker}.csv', index=False)
        

        vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary=lda_model.id2word)
        pyLDAvis.save_html(vis, f'lda_{speaker}.html')
    
    return None


def topicModellingPerSpeaker(numTopics: int, stopwords: set, iterations: int = 2_000) -> None:
    metadataDF: pd.DataFrame = loadMetadata()

    bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
    trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))

    # Combine all documents for all speakers
    combined_documents = []
    speakerDocuments = {}
    MAX_TOKENS = 33_800

    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]
        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
        tokenTbl = tokenTbl[~tokenTbl['LEMMA'].isin(stopwords)]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns
        tokenTbl = tokenTbl[['TEXT_ID', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)

        print(speaker, len(tokenTbl), tokenTbl['TEXT_ID'].max())

        documents = tokenTbl.loc[:MAX_TOKENS, :].groupby('TEXT_ID')['LEMMA'].apply(list).values

        documents = bigram[documents]
        documents = trigram[documents]

        combined_documents.extend(documents)
        speakerDocuments[speaker] = documents


    # topic modelling
    dictionary = gensim.corpora.Dictionary(combined_documents)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10_000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(combined_documents, ncols=80)]

    # run LDA
    lda_model = models.ldamodel.LdaModel(corpus=bow_corpus,
                                        id2word=dictionary,
                                        num_topics=numTopics,
                                        random_state=100,
                                        update_every=1,
                                        chunksize=200, #1000,
                                        passes=20, #10,
                                        alpha='auto', #'symmetric',
                                        iterations=500, #iterations,
                                        per_word_topics=True)


    # Visualize topics for each speaker
    for speaker in metadataDF['speaker'].unique():

        speaker_bow_corpus = [dictionary.doc2bow(doc) for doc in speakerDocuments[speaker]]
        vis = pyLDAvis.gensim_models.prepare(lda_model, speaker_bow_corpus, dictionary=lda_model.id2word, mds='tsne')
        pyLDAvis.save_html(vis, f'_lda_{speaker}.html')

    return None


def topicModellingPerPeriod(numTopics: int, stopwords: set, iterations: int = 2_000) -> None:
    metadataDF: pd.DataFrame = loadMetadata()

    bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
    trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))

    # Combine all documents for all speakers
    MAX_TOKENS = 33_800

    for period in metadataDF['period'].unique():
        combinedDocuments = []
        speakerDocuments = {}

        periodDF = metadataDF[metadataDF['period'] == period]

        for speaker in periodDF['speaker'].unique():

            df = periodDF[periodDF['speaker'] == speaker]
            tokenTbl: pd.DataFrame = concatTables(df)

            tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
            tokenTbl = tokenTbl[~tokenTbl['LEMMA'].isin(stopwords)]
            tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns
            tokenTbl = tokenTbl[['TEXT_ID', 'LEMMA']]
            tokenTbl.reset_index(inplace=True)

            documents = tokenTbl.loc[:MAX_TOKENS, :].groupby('TEXT_ID')['LEMMA'].apply(list).values

            documents = bigram[documents]
            documents = trigram[documents]

            combinedDocuments.extend(documents)
            speakerDocuments[speaker] = documents


        # topic modelling
        dictionary = gensim.corpora.Dictionary(combinedDocuments)
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10_000)
        bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(combinedDocuments, ncols=80)]

        # run LDA
        lda_model = models.ldamodel.LdaModel(corpus=bow_corpus,
                                            id2word=dictionary,
                                            num_topics=numTopics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=200, #1000,
                                            passes=20, #10,
                                            alpha='auto', #'symmetric',
                                            iterations=500, #iterations,
                                            per_word_topics=True)


        # Visualize topics for each speaker
        for speaker in periodDF['speaker'].unique():

            speaker_bow_corpus = [dictionary.doc2bow(doc) for doc in speakerDocuments[speaker]]
            vis = pyLDAvis.gensim_models.prepare(lda_model, speaker_bow_corpus, dictionary=lda_model.id2word, mds='tsne')
            pyLDAvis.save_html(vis, f'LDA_{period}_{speaker}.html')

    return None


def topicModellingPerPeriod2(numTopics: int, stopwords: set, iterations: int = 2_000, LOAD: bool = True) -> None:
    metadataDF: pd.DataFrame = loadMetadata()

    bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
    trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))

    # Combine all documents for all speakers
    MAX_TOKENS = 33_800
    
    # Train a global LDA model across all periods and speakers

    if not LOAD:
        global_combined_documents = []
        global_speaker_documents = {}

        for period in metadataDF['period'].unique():
            periodDF = metadataDF[metadataDF['period'] == period]
            for speaker in periodDF['speaker'].unique():
                df = periodDF[periodDF['speaker'] == speaker]
                tokenTbl: pd.DataFrame = concatTables(df)

                tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
                tokenTbl = tokenTbl[~tokenTbl['LEMMA'].isin(stopwords)]
                tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]
                tokenTbl = tokenTbl[['TEXT_ID', 'LEMMA']]
                tokenTbl.reset_index(inplace=True)

                documents = tokenTbl.loc[:MAX_TOKENS, :].groupby('TEXT_ID')['LEMMA'].apply(list).values
                documents = bigram[documents]
                documents = trigram[documents]

                global_combined_documents.extend(documents)
                global_speaker_documents[(period, speaker)] = documents

        with open('analysis/topicModelling/global_speaker_documents.pkl', 'wb') as f:
            pickle.dump(global_speaker_documents, f)

        with open('analysis/topicModelling/global_combined_documents.pkl', 'wb') as f:
            pickle.dump(global_combined_documents, f)

    else:
        with open('analysis/topicModelling/global_speaker_documents.pkl', 'rb') as f:
            global_speaker_documents = pickle.load(f)
        with open('analysis/topicModelling/global_combined_documents.pkl', 'rb') as f:
            global_combined_documents = pickle.load(f)


    # Train the global LDA model
    dictionary = gensim.corpora.Dictionary(global_combined_documents)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10_000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tqdm(global_combined_documents, ncols=80)]

    for numTopics in range(5, 15):
        lda_model = models.ldamodel.LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=numTopics,
            random_state=100,
            update_every=1,
            chunksize=200,
            passes=20,
            alpha='auto',
            iterations=1_000,
            per_word_topics=True
        )

        coherence_model = models.CoherenceModel(model=lda_model, texts=global_combined_documents, dictionary=dictionary, coherence='c_v')
        print("Topics", numTopics, "Coherence Score:", coherence_model.get_coherence())


    # Visualize topics for individual speakers
    # for (period, speaker), documents in global_speaker_documents.items():
    #     speaker_bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    #     vis = pyLDAvis.gensim_models.prepare(lda_model, speaker_bow_corpus, dictionary=lda_model.id2word, mds='tsne')
    #     pyLDAvis.save_html(vis, f'LDA_{period}_{speaker}.html')


    # Get topic distributions for each speaker
    # for (period, speaker), documents in global_speaker_documents.items():
    #     speaker_bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    #     topic_distribution = lda_model.get_document_topics(speaker_bow_corpus, minimum_probability=0.0)
    #     avg_topic_distribution = np.mean([[topic[1] for topic in doc] for doc in topic_distribution], axis=0)
    #     print(f"Period: {period}, Speaker: {speaker}, Topic Distribution: {avg_topic_distribution}")



    return None


def ensembleTopicModelling(numTopics:int, stopwords:set, iterations:int = 2_000) -> None:

    metadataDF: pd.DataFrame = loadMetadata()

    for speaker in metadataDF['speaker'].unique():

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords)]
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


def authorTopicModelling(numTopics:int, stopwords:list[str], LOAD: bool = False) -> None:
    import plotly.express as px

    path = Path('corpus/tables')
    # gensim.models.AuthorTopicModel()

    bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
    trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))

    metadataDF = loadMetadata()
    author2doc = defaultdict(list)
    texts = []

    if not LOAD:
        for speaker in metadataDF['speaker'].unique():
            df = metadataDF[metadataDF['speaker'] == speaker]

            for file in df['linkTables']:
                tablePath = path / file
                table = pd.read_csv(tablePath)
                tokensRaw = filterTokens(table, stopwords)

                tokens = bigram[tokensRaw]
                tokens = trigram[tokens]

                texts.append(tokens)
                author2doc[speaker].append(len(texts) - 1)


        with open('analysis/topicModelling/authorTexts.pkl', 'wb') as f:
            pickle.dump(texts, f)
        
        with open('analysis/topicModelling/author2doc.pkl', 'wb') as f:
            pickle.dump(author2doc, f)
    
    else:
        with open('analysis/topicModelling/authorTexts.pkl', 'rb') as f:
            texts = pickle.load(f)
        
        with open('analysis/topicModelling/author2doc.pkl', 'rb') as f:
            author2doc = pickle.load(f)


    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.4, keep_n=10_000)
    corpus = [dictionary.doc2bow(text) for text in texts]



    model = gensim.models.AuthorTopicModel(
        corpus=corpus,
        num_topics=numTopics,
        id2word=dictionary,
        author2doc=author2doc,
        update_every=1,
        chunksize=100,
        iterations=250,
        passes=15,
        alpha=0.1,
        eta=0.01 #'auto'
    )

    # Step 4: Analyze Topics per Speaker
    # Get the topics associated with each speaker
    df = pd.DataFrame(columns=['speaker', 'topic', 'percentage'])

    for author in model.author2doc.keys():

        authorTopics = model.get_author_topics(author, minimum_probability=0.0)

        for topic in authorTopics:
            df.loc[len(df)] = [author, *topic]
    
    group = df[['topic', 'percentage']].groupby('topic').sum()
    group.sort_values(by=['percentage'], ascending=False, inplace=True)

        

    df['relPercentage'] = df['percentage'] / df['speaker'].nunique()
    fig = px.bar(df, 'topic', 'relPercentage', color='speaker', range_y=(0,1))
    fig.show()


    coherence_model = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    print("Topics", numTopics, "Coherence Score:", coherence_model.get_coherence())
    logPerplexity = model.log_perplexity(corpus, chunk_doc_idx=list(range(len(corpus))))
    print(f"Model Perplexity: perplexity = {math.e ** logPerplexity}")
    print()

    for index, row in group.iterrows():
        topics = model.show_topic(index)
        print(f'topic {index}:')
        print([tp[0] for tp in topics[:10]])
        print()
        if index == 3:
            break
    
    # Step 5: Examine a specific topic
    # Get the top words for topic 0
    # print("Top words for topic 0:")
    # print(model.print_topics())
    return None


def authorTopicModellingPeriod(numTopics:int, stopwords:set[str], save:bool = False) -> None:
    
    path = Path('corpus/tables')
    savepath = Path('analysis/topicModelling/topicsPerPeriod')
    # gensim.models.AuthorTopicModel()

    bigram = models.Phrases.load(str(Path('analysis/topicModelling/bigram.model')))
    trigram = models.Phrases.load(str(Path('analysis/topicModelling/trigram.model')))

    metadataDF = loadMetadata()

    extendedStopwords = {'comment', 'self', 'brown', 'lot', 'story', 'tonight', 'nomination', 'nominee', 
                         'page', 'comment', 'desk', 'door', 'walz', 'michelle'
                         }

    for period in metadataDF['period'].unique():
        periodDF = metadataDF[metadataDF['period'] == period]

        author2doc = defaultdict(list)
        texts = []

        savepathPeriod = savepath / f'period{period}'
        savepathPeriod.mkdir(exist_ok=True)

        print(f'Period {period}, speaker {periodDF['speaker'].unique()}')

        docLimit = getDocLimit(periodDF)  # balance doc amount

        for speaker in periodDF['speaker'].unique():
            df = periodDF[periodDF['speaker'] == speaker].reset_index()

            speakerStopwords = set()
            for speaker_ in periodDF['speaker'].unique():
                speakerStopwords.add(speaker_.lower().replace(' ', '_'))
                speakerStopwords |= set(speaker_.lower().split())

            for file in df.loc[:docLimit, 'linkTables']:
                tablePath = path / file
                table = pd.read_csv(tablePath)
                tokensRaw = filterTokens(table, stopwords | speakerStopwords | extendedStopwords)

                tokens = bigram[tokensRaw]
                tokens = trigram[tokens]

                texts.append(tokens)
                author2doc[speaker].append(len(texts) - 1)


        dictionary = gensim.corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10_000)
        corpus = [dictionary.doc2bow(text) for text in texts]


        c_v = 0.0
        while c_v < 0.25:
            model, c_v, resultString = trainAuthorTopicModel(corpus, dictionary, author2doc, texts, numTopics)
            # print(f'{c_v=}')
            print(resultString)

        fig = plotBarChart(model, period)
        if save:
            model.save(str(savepathPeriod / f'AuthorTopicModel_period{period}.model'))

            with open(savepathPeriod / f'AuthorTopicModel_period{period}.txt', 'w') as f:
                f.write(resultString)

            fig.write_image(str(savepathPeriod / f'AuthorTopicModel_period{period}.png'), height=800, width=1200)
        else:
            fig.show()
        break
    # Step 5: Examine a specific topic
    # Get the top words for topic 0
    # print("Top words for topic 0:")
    # print(model.print_topics())
    return None

def trainAuthorTopicModel(corpus, dictionary, author2doc, texts, numTopics) -> tuple[float, str]:
    model = gensim.models.AuthorTopicModel(
        corpus=corpus,
        num_topics=numTopics,
        id2word=dictionary,
        author2doc=author2doc,
        update_every=1,
        chunksize=100,
        iterations=10_000,
        passes=30,
        alpha=0.55,
        eta=0.1,
    )

    # fig = plotBarChart(model, period)
    # fig.show()
    c_v, resultString = evaluateTopicModel(model, texts, corpus, dictionary)
    # print(f'{c_v=}')
    return model, c_v, resultString


def plotBarChart(model, period):
    
    df = pd.DataFrame(columns=['speaker', 'topic', 'percentage'])

    for author in model.author2doc.keys():

        authorTopics = model.get_author_topics(author, minimum_probability=0.0)

        for topic in authorTopics:
            df.loc[len(df)] = [author, *topic]
        
    df['relPercentage'] = df['percentage'] / df['speaker'].nunique()
    fig = px.bar(df, 'topic', 'relPercentage', color='speaker', range_y=(0,1), color_discrete_map=COLOR_MAP,
             title=f'Topic Distribution Election Period {period}')

    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    return fig


def evaluateTopicModel(model:models, texts:list[str], corpus, dictionary) -> float:
      
    ## print statistics ##
    result = ['-' * 80]
    result.append('Topic Model Evaluation')
    result = ['-' * 80]

    result.append(f'Hyperparameter of {model.__class__.__qualname__}:')
    result.append(f'Update every: {model.update_every:>4} | Chunksize: {model.chunksize}')
    result.append(f'Iterations: {model.iterations:>6} | Passes: {model.passes}')
    result.append(f'Alpha: {float(model.alpha[0]):>11} | Eta: {float(model.eta[0])}')
    result.append('')

    coherence_c_v = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_u_mass = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
    logPerplexity = model.log_perplexity(corpus, chunk_doc_idx=list(range(len(corpus))))

    result.append(f'Coherence Score c_v: {coherence_c_v.get_coherence():1.6f} (aim for >0.5), u_mass: {coherence_u_mass.get_coherence():1.6f} (aim for >-2)')
    result.append(f'Perplexity: {math.e ** logPerplexity:1.6f}')

    authorTopics = defaultdict(dict)
    for author in model.author2doc.keys():
        topics = model.get_author_topics(author)
        for tp in topics:
            authorTopics[author][tp[0]] = tp[1]

    for ntopic, (c_v, u_mass) in enumerate(zip(coherence_c_v.get_coherence_per_topic(), coherence_u_mass.get_coherence_per_topic())):

        topics = model.show_topic(ntopic)
        authorTopicsString = ' | '.join([f'{author}: {value.get(ntopic, 0.0):1.6f}' for author, value in authorTopics.items()])

        result.append('')
        result.append(f'Topic {ntopic} (c_v = {c_v:1.6f}, u_mass = {u_mass:1.6f}):')
        result.append(f'  {authorTopicsString}')
        result.append(f'  - {", ".join([tp[0] for tp in topics])}')
    
    result.append('')
    result.append('-' * 80)

    resultString = '\n'.join(result)
    # print(resultString)
    return coherence_c_v.get_coherence(), resultString


def getDocLimit(periodDF:pd.DataFrame) -> int:
    docLength = []
    for speaker in periodDF['speaker'].unique():
        docLength.append(len(periodDF[periodDF['speaker'] == speaker]))
    
    return min(docLength)


def filterTokens(tokenTbl:pd.DataFrame, stopwords:Iterable) -> list[str]:

    tokenTbl = tokenTbl[~tokenTbl['TOKEN'].str.lower().isin(stopwords)]
    tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

    return tokenTbl['LEMMA'].str.lower().tolist()


def trainBigrams(stopwords:Iterable, save: bool = False, analyze: bool = False) -> None:

    metadataDF: pd.DataFrame = loadMetadata()
    sentences = []

    for speaker in tqdm(metadataDF['speaker'].unique(), total=len(metadataDF['speaker'].unique()), ncols=80, desc='Loading data'):

        ## data access & preprocessing
        df = metadataDF[metadataDF['speaker'] == speaker]

        tokenTbl: pd.DataFrame = concatTables(df)

        tokenTbl = tokenTbl[~tokenTbl['TOKEN'].isin(stopwords)]
        tokenTbl = tokenTbl[tokenTbl['POS'].isin(['NOUN', 'PROPN'])]  # use only nouns

        tokenTbl = tokenTbl[['TEXT_ID', 'ID', 'SENTENCE_ID', 'TOKEN_ID', 'TOKEN', 'LEMMA']]
        tokenTbl.reset_index(inplace=True)
        tokenTbl.sort_values(by=['TEXT_ID', 'ID'])

        tokenTbl['LEMMA'] = tokenTbl['LEMMA'].str.lower()
        documents = tokenTbl[['TEXT_ID', 'SENTENCE_ID', 'LEMMA']].groupby(['TEXT_ID', 'SENTENCE_ID']).agg({'LEMMA': list})

        sentences.extend(documents['LEMMA'].values)


    bigramPhrases = models.Phrases(sentences,
                                   min_count=30,
                                   threshold=85,)
    trigramPhrases = models.Phrases(bigramPhrases[sentences],
                                   min_count=20,
                                   threshold=95,)
    
    bigram = models.phrases.Phraser(bigramPhrases)
    trigram = models.phrases.Phraser(trigramPhrases)


    if save:
        trigram.save(str(Path('analysis/topicModelling/trigram.model')))
        bigram.save(str(Path('analysis/topicModelling/bigram.model')))

    if analyze:
        analyzeNGrams([bigram, trigram], sentences)

    return None


def analyzeNGrams(models:list[models.Phrases], sentences: list[str], n_most: int = 20) -> None:
    nameSuffixes = ['Bi', 'Tri', 'Four', 'Five', 'Six', 'Seven']

    nGramNames = [f'{suffix}gram' for suffix in nameSuffixes[:len(models)]]
    nGramPattern = {name: re.compile(fr'.+{"_.+" * i}') for i, name in enumerate(nGramNames, 1)}
    result = []

    for model, (name, pattern) in zip(models, nGramPattern.items()):
        dataNGram = [model[doc] for doc in sentences]
        nGrams = [word for sentence in dataNGram for word in sentence if pattern.match(word)]
        sentences = dataNGram

        counter = Counter(nGrams)
        result.append(f'Found {len(counter)} {name}, {n_most} most frequent:')
        for value, count in counter.most_common(n_most):
            result.append(f' - {value:<20} - {count:>4}')
        result.append('\n')
    
    result = '\n'.join(result)
    print(result)
    with open(Path('analysis/topicModelling/NGramExamples.txt'), 'w') as f:
        f.write(result)

    return None

        
def getTopicWords(lda, numWords: int = 10) -> pd.DataFrame:
    df = pd.DataFrame(columns=['Topic'] + [f'Word{i:02d}' for i in range(numWords)])

    for topic in lda.show_topics(numWords):
        wordString = topic[1]
        words = [w.split('*')[1].strip('"') for w in wordString.split(' + ')]
        
        df.loc[len(df)] = [topic[0]] + words
    
    return df


def getStopwords() -> set[str]:

    from string import punctuation
    from gensim.parsing.preprocessing import STOPWORDS

    punctuation = set(punctuation)
    punctuation |= {'--', '...', r'\u2013', r'\u2014', 'dr.', '–', 'mr', 'hi', 'mr.', 'cheer', 'applause', 'sir', 'laughter',
                    'audience', 'boo', 'booo', 'laughs', '--audience', 'â€'}
    stopwords = STOPWORDS | punctuation

    return stopwords


def guessTopicName() -> None:
    from wn import Wordnet
    import itertools

    path = 'analysis/topicModelling/topics/topicsBarack Obama.csv'

    wordDF = pd.read_csv(path)
    
    wordnet = Wordnet(lang='en')


    print(wordDF.iloc[0,1])
    print(wordDF.iloc[2,1])
    synsets = wordnet.synsets(wordDF.iloc[0,1])
    for (synsetA, synsetB) in itertools.permutations(synsets, 2):
        common = synsetA.lowest_common_hypernyms(synsetB)
        print(synsetA.lemmas(), synsetB.lemmas(), *[syn.lemmas() for syn in common])
        print()
        print(synsetA.hypernyms()[0].lemmas())
        print(synsetB.hypernyms()[0].lemmas())




if __name__ == '__main__':

    numTopics = 6  # maybe less
    stopwords = getStopwords()

    # trainBigrams(stopwords, analyze=True, save=False)
    # topicModellingPerSpeaker(numTopics, stopwords)
    # topicModellingPerPeriod2(numTopics, stopwords, LOAD=True)
    # authorTopicModelling(numTopics, stopwords, LOAD=False)
    # authorTopicModellingPeriod(numTopics, stopwords)

    # ensembleTopicModelling(numTopics)

    # guessTopicName()  # experimental!
