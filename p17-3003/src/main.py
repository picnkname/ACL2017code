import CNNModel
import DataHandle
import numpy as np
from tflearn.data_utils import to_categorical
from sklearn.model_selection import train_test_split
import os
# only use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    dim = 12
    print('loading data')
    raw_de, raw_en, raw_gold = DataHandle.loadRawData('../data/bucc2017/de-en/de-en.training.en',
                                                      '../data/bucc2017/de-en/de-en.training.de',
                                                      '../data/bucc2017/de-en/de-en.training.gold')
    Data = DataHandle.getPositiveNegative(raw_de, raw_en, raw_gold)
    De = DataHandle.loadWordVecs('../data/unsup.128.de')
    En = DataHandle.loadWordVecs('../data/unsup.128.en')
    train_x, train_y = DataHandle.convertDataToMatrix(Data, De, En, dim)
    ### count the data num
    count_0 = 0
    count_1= 0
    for item in train_y:
        if item == 0:
            count_0 += 1
        else:
            count_1 += 1
    print('sample num:')
    print('total : %d' % (count_0 + count_1))
    print('positive : %d' % count_1)
    print('negative : %d' % count_0)
    ###
    train_y = to_categorical(train_y, nb_classes=2)
    # split train and test set
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,train_size=0.8)
    model = CNNModel.build_model(input_size=dim, input_channels=1)
    model.fit({'input': train_x}, {'target': train_y}, n_epoch=20,
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=100, show_metric=True, run_id='bilingual')
