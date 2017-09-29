import collections
import nltk

feature_vocab = ['phone', 'computer', 'gps', 'box', 'line', 'button', 'side', 'cord', 'wire', 'keyboard', 'headset',
                 'build', 'line', 'microphone', 'mic', 'volume', 'resolution', 'picture', 'sound', 'port', 'battery',
                 'unit', 'screen', 'price', 'cost', 'notifications', 'databases', 'database', 'screen','card','tablet',
                 'nook','books','route','images','image','pictures','focus','lens','flash','mount']


def findFrequent(taglist, total):
    freqlist = []
    counter = collections.Counter(taglist)
    # print(counter)
    for tag in counter.keys():
        number = counter.get(tag)
        support = number / total
        # print(support)
        if feature_vocab.__contains__(tag[0].lower()):
            if support >= 0.0025:
                freqlist.append(tag)
    if len(freqlist) == 0:
        for tag in counter.keys():
            number = counter.get(tag)
            support = number / total
            # print(support)
            if feature_vocab.__contains__(tag[0]):
                if support >= 0.0015:
                    freqlist.append(tag)

    return freqlist


def extractOpinion(featurelist, taglist):
    opinionWords = []
    previous = 0
    nxt = 1
    length = len(taglist)
    for tuple in taglist:
        if tuple[1] == 'NN' or tuple[1] == 'NNS':
            if any(t[0] == tuple[0] for t in featurelist):
                if taglist[previous][1] == 'JJ' or taglist[previous][1] == 'JJR' or taglist[previous][1] == 'JJS':
                    adjs = []
                    newt = tuple[0], tuple[1], taglist[previous]
                    #   opinionWords.append(newt)
                    adjs.append(newt)
                    if (taglist[nxt][1] == 'JJ' or taglist[nxt][1] == 'JJR' or taglist[nxt][1] == 'JJS') and (
                                    (nxt + 1) < (length - 1)
                            and (taglist[nxt + 1][1] != 'NN' or taglist[nxt + 1][1] != 'NNS')):
                        newt = tuple[0], tuple[1], taglist[nxt]
                        adjs.append(newt)
                    opinionWords.append(newt)
                else:
                    if taglist[nxt][1] == 'JJ' or taglist[nxt][1] == 'JJR' or taglist[nxt][1] == 'JJS':
                        newt = tuple[0], tuple[1], taglist[nxt]
                        opinionWords.append(newt)
        if previous < (length - 1):
            previous = previous + 1
        if nxt < (length - 1):
            nxt = nxt + 1
    return opinionWords


def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]


def infrequentFeatures(reviewlist, freqlist):
    opinions = read_words('sentiment_words.txt')
    # print(opinions)
    for sentence in reviewlist:
        text = sentence.split('.')
        for line in text:
            # print(line)
            flag = 0
            token = nltk.word_tokenize(line)  # tokenize each word
            for word in token:
                # print(word)
                if any(word == t[0] for t in freqlist):
                    flag = 1
                    break
            if flag == 0:
                for word in token:
                    if opinions.__contains__(word):
                        tags = nltk.pos_tag(token)
                        # print(tags)
                        for tag in tags:
                            if tag[0] == 'NN' or tag[0] == 'NNS':
                                #    print(tag)
                                freqlist.append(tag)

    return freqlist
