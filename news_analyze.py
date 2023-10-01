import pandas as pd
import json

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

def final_data():

    # Load JSON data
    with open('Trading_News.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Example: Analyzing sentiment of the first string in the JSON data
    key, text = list(data.items())[3]
    BHP = [i for i in range(50,79)]
    CSL = [i for i in range(108,119)]
    RIO = [i for i in range(149,170)]
    CBA = [i for i in range(213,239)]
    WOW = [i for i in range(257,276)]
    WES = [i for i in range(299,319)]
    TLS = [i for i in range(332,352)]
    AMC = [i for i in range(365,379)]
    BXB = [i for i in range(385,392)]
    FPH = [i for i in range(398,404)]

    BHP_list = []
    CSL_list = []
    RIO_list = []
    CBA_list = []
    WOW_list = []
    WES_list = []
    TLS_list = []
    AMC_list = []
    BXB_list = []
    FPH_list = []

    for key, value in text.items():
        # print(key)
        sentiment, probabilities = analyze_sentiment(text[key])
        # print(text[key])
        # print(f"Predicted Sentiment: {sentiment}")
        # print(f"Probabilities: {probabilities}")
        if int(key) in BHP:
            BHP_list.append(sentiment)
        if int(key) in CSL:
            CSL_list.append(sentiment)
        if int(key) in CBA:
            CBA_list.append(sentiment)
        if int(key) in RIO:
            RIO_list.append(sentiment)
        if int(key) in WOW:
            WOW_list.append(sentiment)
        if int(key) in WES:
            WES_list.append(sentiment)
        if int(key) in TLS:
            TLS_list.append(sentiment)
        if int(key) in AMC:
            AMC_list.append(sentiment)
        if int(key) in BXB:
            BXB_list.append(sentiment)
        if int(key) in FPH:
            FPH_list.append(sentiment)
    # 定义目标长度
    target_length = 42

    # 对每家公司的列表进行插值
    BHP_list = linear_interpolation(BHP_list, target_length)[::-1]
    CSL_list = linear_interpolation(CSL_list, target_length)[::-1]
    RIO_list = linear_interpolation(RIO_list, target_length)[::-1]
    CBA_list = linear_interpolation(CBA_list, target_length)[::-1]
    WOW_list = linear_interpolation(WOW_list, target_length)[::-1]
    WES_list = linear_interpolation(WES_list, target_length)[::-1]
    TLS_list = linear_interpolation(TLS_list, target_length)[::-1]
    AMC_list = linear_interpolation(AMC_list, target_length)[::-1]
    BXB_list = linear_interpolation(BXB_list, target_length)[::-1]
    FPH_list = linear_interpolation(FPH_list, target_length)[::-1]

    # 将10个公司的列表组合成一个矩阵
    matrix = np.array([BHP_list, CSL_list, RIO_list, CBA_list, WOW_list, WES_list, TLS_list, AMC_list, BXB_list, FPH_list])

    print(matrix)



def analyze_sentiment(text):
    # Load Pre-trained BERT model and tokenizer
    model_name = 'ProsusAI/finbert'
    # 指定您存放模型文件的本地目录
    model_directory = 'E:\workspace\CANOE_test\investment_portfolio\\finbert_lib'
    model = BertForSequenceClassification.from_pretrained(model_directory)
    tokenizer = BertTokenizer.from_pretrained(model_directory)
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt')

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute softmax to get probabilities
    probs = softmax(outputs.logits, dim=-1)

    # Get the predicted sentiment
    _, predicted = torch.max(probs, dim=-1)

    return predicted.item(), probs.numpy()


def linear_interpolation(lst, target_length):
    # 当前列表的长度
    original_length = len(lst)

    # 如果原列表长度已经等于目标长度，则直接返回原列表
    if original_length == target_length:
        return lst

    # 初始化插值后的列表
    interpolated_list = []

    # 计算插值的步长
    step = (original_length - 1) / (target_length - 1)

    for i in range(target_length):
        # 找到插值位置的两个邻近点的索引
        lower_index = int(i * step)
        upper_index = min(lower_index + 1, original_length - 1)

        # 如果两个索引相等，则直接取该点的值
        if lower_index == upper_index:
            interpolated_list.append(lst[lower_index])
        else:
            # 如果不相等，则进行线性插值
            ratio = (i * step) - lower_index
            value = (1 - ratio) * lst[lower_index] + ratio * lst[upper_index]
            interpolated_list.append(value)

    return interpolated_list

final_data()

