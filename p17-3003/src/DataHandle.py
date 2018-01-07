import pickle
import math
import random
import numpy as np
import sklearn.metrics as sk
import os


def loadWordVecs(file):
    f = open(file, 'r', encoding='utf8')
    cnt = 0
    wv = {}
    for line in f:
        if cnt > 0:
            line = line.split()
            wv[line[0]] = np.array([np.float(j) for j in line[1:]]).reshape(1, -1)
        cnt = cnt + 1
    return wv


def loadRawData(file_en, file_de, file_gold):
    raw_en = {}
    with open(file_en, 'r', encoding='utf8') as f:
        for line in f:
            idx = line.find('\t')
            sentence = line[idx+1:].strip('\n')
            sentence_id = line[:idx]
            raw_en[sentence_id] = sentence
    raw_de = {}
    with open(file_de, 'r', encoding='utf8') as f:
        for line in f:
            idx = line.find('\t')
            sentence = line[idx+1:].strip('\n')
            sentence_id = line[:idx]
            raw_de[sentence_id] = sentence
    raw_gold = []
    with open(file_gold, 'r', encoding='utf8') as f:
        for line in f:
            line = line.split()
            raw_gold.append((line[0], line[1]))
    return raw_de, raw_en, raw_gold


def random_bin(p):
    l = np.random.randint(0, 1000)
    if l <= int(1000 * p):
        return True
    return False


def getPositiveNegative(raw_de, raw_en, raw_gold):
    Data = []
    # get the positive sample according to the gold file
    for g in raw_gold:
        d = []
        d.append(raw_de[g[0]])
        d.append(raw_en[g[1]])
        d.append(1) # label
        Data.append(d)
    list_en = list(raw_en.keys())
    len_en = len(list_en)
    len_gold = len(set([item[0] for item in raw_gold]))
    p = 0.03
    for idx, j in enumerate(raw_de.keys()):
        if random_bin(p * (len(raw_gold)) / len_gold):
            d = []
            val = list_en[np.random.randint(0, len_en)]
            try:
                if raw_gold[j][val]:
                    pass
            except:
                d.append(raw_de[j])
                d.append(raw_en[val])
                d.append(0)
                Data.append(d)
    # shuffle
    random.shuffle(Data)
    return Data


def convertDataToMatrix(Data, De, En, dim=15):
    mat_data = []
    label = []
    for idx in range(len(Data)):
        sen1 = Data[idx][0]
        sen2 = Data[idx][1]

        word_list1 = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in sen1.split()]
        word_list2 = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in sen2.split()]

        vec1 = []
        for word in word_list1:
            try:
                vec1.append(De[word])
            except:
                # print('%s not in vocab' % word)
                pass
        if (len(vec1) == 0):
            vec1.append(De['.'])
        vec2 = []
        for word in word_list2:
            try:
                vec2.append(En[word])
            except:
                # print('%s not in vocab' % word)
                pass
        if (len(vec2) == 0):
            vec2.append(En['.'])
        mat = getSimilarityMatrix(vec1, vec2)
        pooled_mat = getDynamicPooledMatrix(mat, dim)
        len_mean = (len(vec1) + len(vec2)) / 2.0
        if len_mean >=12 and len_mean <=15:
            mat_data.append(pooled_mat)
            label.append(Data[idx][2])
    assert len(mat_data) == len(label), 'the length should be same'

    return np.array(mat_data), np.array(label)


def getSimilarityMatrix(vec1, vec2, metric='cos'):
    mat = np.zeros((len(vec1), len(vec2)))
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            if metric == 'cos':
                mat[i][j] = sk.pairwise.cosine_similarity(vec1[i], vec2[j])
            elif metric == 'euclid':
                mat[i][j] = sk.pairwise.euclidean_distances(vec1[i], vec2[j])
    return mat


def getDynamicPooledMatrix(mat, dim=15, type='max'):
    l1 = np.shape(mat)[0]
    l2 = np.shape(mat)[1]

    wide_x = wide_y = 1

    if l1 < dim:
        wide_x = int(math.ceil(dim * 1.0 / l1))

    if l2 < dim:
        wide_y = int(math.ceil(dim * 1.0 / l2))

    Mat = np.zeros(np.shape(mat))
    Mat = mat
    for i in range(1, wide_x):
        Mat = np.append(Mat, mat, axis=0)

    Matt = np.zeros(np.shape(Mat))
    Matt = Mat
    for i in range(1, wide_y):
        Matt = np.append(Matt, Mat, axis=1)

    dim1 = np.shape(Matt)[0] * 1.0
    dim2 = np.shape(Matt)[1] * 1.0

    chunk_size1 = int(dim1 / dim)
    chunk_size2 = int(dim2 / dim)

    pooled_mat = np.zeros((dim, dim))

    for i in range(dim * chunk_size1, int(dim1)):
        for j in range(0, int(dim2)):
            if type == 'max':
                Matt[i - (int(dim1) - dim * chunk_size1)][j] = max(Matt[i - (int(dim1) - dim * chunk_size1)][j],
                                                                   Matt[i][j])
            elif type == 'min':
                Matt[i - (int(dim1) - dim * chunk_size1)][j] = min(Matt[i - (int(dim1) - dim * chunk_size1)][j],
                                                                   Matt[i][j])

    for j in range(dim * chunk_size2, int(dim2)):
        for i in range(0, int(dim1)):
            if type == 'max':
                Matt[i][j - (int(dim2) - dim * chunk_size2)] = max(Matt[i][j - (int(dim2) - dim * chunk_size2)],
                                                                   Matt[i][j])
            elif type == 'min':
                Matt[i][j - (int(dim2) - dim * chunk_size2)] = min(Matt[i][j - (int(dim2) - dim * chunk_size2)],
                                                                   Matt[i][j])

    for i in range(0, dim):
        for j in range(0, dim):
            val = Matt[i * chunk_size1][j * chunk_size2]
            for u in range(i * chunk_size1, (i + 1) * chunk_size1):
                for v in range(j * chunk_size2, (j + 1) * chunk_size2):
                    if type == 'max':
                        val = max(val, Matt[u][v])
                    elif type == 'min':
                        val = min(val, Matt[u][v])

            pooled_mat[i][j] = val
    pooled_mat = np.reshape(pooled_mat, [dim,dim,-1])
    return pooled_mat


def load_Word_Vecs_for_Data_Bucket():
    filed = open('../data/bucc2017/de-en/train_valid_test/data_de_en_vecs_bucketed', 'rb')
    data = pickle.load(filed, encoding='bytes')
    filed.close()
    train_X = data[0]
    train_Y = data[1]
    valid_X = data[2]
    valid_Y = data[3]
    test_X = data[4]
    test_Y = data[5]
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


if __name__ == '__main__':
    # raw_de, raw_en, raw_gold = loadRawData('../data/bucc2017/de-en/de-en.training.en',
    #                                        '../data/bucc2017/de-en/de-en.training.de',
    #                                        '../data/bucc2017/de-en/de-en.training.gold')
    # Data = getPositiveNegative(raw_de,raw_en, raw_gold)
    # De = loadWordVecs('../data/unsup.128.de')
    # En = loadWordVecs('../data/unsup.128.en')
    # vocab = De
    # for item in En.keys():
    #     vocab[item] = En[item]
    # train_x, train_y = convertDataToMatrix(Data, vocab=vocab)
    # print(len(train_x))
    # print(len(train_y))
    # load_Word_Vecs_for_Data_Bucket()
    pass

