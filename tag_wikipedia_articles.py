__author__ = 'rajareddykarri'

import wikipedia
import gensim

electrical_links = wikipedia.page("Index_of_electrical_engineering_articles").links
dictionary = gensim.corpora.Dictionary()
print "No. of keys at start ", dictionary.keys().__len__()

# Pass 1: Prepare a dictionary
for link in electrical_links:
    try:
        page = wikipedia.page(link)
    except:
        continue
    title = gensim.parsing.preprocess_string(page.title)
    content = gensim.parsing.preprocess_string(page.content)

    dictionary.add_documents([title, content])

print "Prepared Dictionary, No. of keys after addition of data ", dictionary.keys().__len__()

# Pass 2: Process topics
lda = gensim.models.ldamodel.LdaModel(corpus=None, id2word=dictionary, num_topics=30, update_every=1, chunksize=1, passes=2)

for link in electrical_links:
    try:
        page = wikipedia.page(link)
    except:
        continue
    title = gensim.parsing.preprocess_string(page.title)
    content = gensim.parsing.preprocess_string(page.content)

    title_bow = dictionary.doc2bow(title)
    content_bow = dictionary.doc2bow(content)

    new_bag_of_words = title_bow + content_bow
    print(content_bow)
    lda.update([content_bow])

    print(link + ": ", lda[new_bag_of_words])

print "done : " + dictionary.__sizeof__()

