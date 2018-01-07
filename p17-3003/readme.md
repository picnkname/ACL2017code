# ACL2017 论文实现

尝试实现论文中的方法。参考了原作者的代码，主要作为练习。使用tflearn来搭建卷积神经网络。

## Paper

paper | BilingualWord Embeddings with Bucketed CNN for Parallel Sentence Extraction

author | Jeenu Grover, Pabitra Mitra

## Pre-requisites

- tensorflow:1.4.0
- tflearn:0.3.2
- numpy1.13.3
- scikit learn:0.19.0
- python:3.6

## Probelm Definition

解决的问题是ACL2017的workshop。

## Data

data文件夹的结构如下所示：

```
data
|--bucc2017
  |--de-en
    |--train_valid_test
      |--data_de_en_vecs_bucketed
    de-en.training.de
    de-en.training.en
    de-en.training.gold
  unsup.128.de
  unsup.128.en
```

## Usage

直接执行src中的main.py即可。

## Attribution / Thanks

论文作者实现的代码地址是：https://github.com/groverjeenu/Bilingual-Word-Embeddings-with-Bucketed-CNN-for-Parallel-Sentence-Extraction

论文中用到的预训练双语词向量，数据的下载地址是：https://nlp.stanford.edu/~lmthang/bivec/

数据集下载地址：https://comparable.limsi.fr/bucc2017/bucc2017-task.html
