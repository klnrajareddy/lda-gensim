__author__ = 'rajareddykarri'

import wikipedia
import gensim

electrical_links = wikipedia.page("Index_of_electrical_engineering_articles").links
dictionary = gensim.corpora.Dictionary()
print "No. of keys at start ", dictionary.keys().__len__()
count = 0

# Pass 1: Prepare a dictionary
for link in electrical_links:
    try:
        page = wikipedia.page(link)
        title = gensim.parsing.preprocess_string(page.title)
        content = gensim.parsing.preprocess_string(page.content)
    except:
        continue
    count += 1
    dictionary.add_documents([title, content])
    print "Adding to dictionary document ", count

print "Prepared Dictionary, No. of keys after addition of data ", dictionary.keys().__len__()

# Pass 2: Process topics
lda = gensim.models.ldamodel.LdaModel(corpus=None, id2word=dictionary, num_topics=30, update_every=1, chunksize=1, passes=2)

count = 0
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
    count += 1
    print(link + ": ", lda[new_bag_of_words])
    print(count)

print "done : " + dictionary.__sizeof__()

