from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pdfplumber
import string
import re
import os
stop_words = set(stopwords.words('english'))
stop_words.update(['-', 'pp', 'reserved', 'rights', 'copyright', 'comm', 'corp', 'doi', 'httpdxdoiorgx', 'eissn', 'wwwcognizantcommunicationcom', 'content', 'context', 'references', 'journal',
                   'authors', 'permission', 'sagepubcoukjournalspermissionsnav', 'e', 'contents', 'sciencedirect', 'homepage', 'wwwelserviercomlocatetourman', 'see', 'httpswwwresearchgatenetpublication', 'publication'])


def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
    return result


def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


def textnormalizer(s):
    s = s.lower()
    s = re.sub("n’t", " not", s)
    s = re.sub("’re", " are", s)
    s = re.sub("’d be", " would be", s)
    s = re.sub("’d", " had", s)
    s = re.sub("’ll", " will", s)
    s = re.sub("won’t", "will not", s)
    s = re.sub("’s", " is", s)
    s = re.sub("’m", " am", s)
    s = s.replace("…", "")
    s = ''.join([i for i in s if not i.isdigit()])
    # str = str.replace(".", "").replace(",", "").replace("...", "").replace("[", "")
    # str = str.replace("//", "").replace("'", "").replace("")

    # OR {key: None for key in string.punctuation}
    table = str.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(table)


files = os.listdir("Papers/2015")
for filename in files:
    print("FILE NAME :", filename)
    doc = ""
    with open(f"Papers/2015/{filename}", "r", encoding="utf-8") as file:
        doc = file.read()
    doc = textnormalizer(doc)
    #print(doc.encode("utf-8").decode("ascii", 'ignore'))
    total_words = doc.split()
    total_word_length = len(total_words)
    print("Total No. of words: ", total_word_length)

    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)
    print("Total Sentences: ", total_sent_len)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    # print(tf_score)

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    # print(idf_score)

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    # print(tf_idf_score)

    try:
        print(get_top_n(tf_idf_score, 20))
    except:
        print("Error")
        continue
