import eikon as ek
import numpy as np
import pandas as pd
import os
import shutil
import zipfile
import datetime
import cufflinks as cf
import configparser as cp
import platform
import pickle
import nltk
nltk.download('stopwords')

from copy import deepcopy
import collections
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
stoppingWordSet = set(stopwords.words('english'))

import tensorflow_hub as hub
import re
import tensorflow as tf
import tensorflow.keras as keras

dataRootPath = r"D:/Eikon_Data/"
dataRootPathNews = r"D:/Eikon_Data/News/"
dataRootPathMarketData = r"D:/Eikon_Data/Market_Data/"
dataRootPathDB = r"D:/Database/"
modelPath = r"D:/python/PROD_Model/"
zipFolderPath = r"D:/Zip_Folder/"
tf_hub_path = r"C:/Users/hc_la/AppData/Local/Temp/tfhub_modules/"
date_format = "%Y-%m-%d"

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

def createFullNameDict():
    df = pd.read_csv(dataRootPathDB+"Underlying_Database/full_name.csv")
    return {u:l.split(",") for u,l in zip(df["undlName"].values, df["full_name_list"].values)}

def getUndlNameList(criterion=""):
    if criterion == "":
        df = pd.read_csv(dataRootPathDB + "Underlying_Database/undlNameList.csv")
        return df.undlName.values
    elif criterion == "HK" or criterion == "AX" or criterion == "SI":
        df = pd.read_csv(dataRootPathDB + "Underlying_Database/undlNameList.csv")
        return [u for u in df.undlName.values if criterion in u]
    else:
        df = pd.read_csv(dataRootPathDB + "Underlying_Database/sector.csv")
        sectorDict = {k:v.split(",") for k, v in zip(df["Cluster"], df["undlNameList"])}
        return sectorDict.get(criterion)

# create undlName full name dict
undlNameFullNameDict = createFullNameDict()

df = pd.read_csv(dataRootPathDB + "Underlying_Database/sector.csv")
undlSectorDict = {}
for cluster, l in zip(df["Cluster"], df["undlNameList"]):
    for u in l.split(","):
        undlSectorDict[u] = cluster

def getSector(undlName):
    return undlSectorDict.get(undlName)

today = datetime.datetime.now()
date_format = "%Y-%m-%d"

def checkFolderExist(path):
    return os.path.isdir(path)

def checkFileExist(path):
    return os.path.isfile(path)

def createFolder(rootPath, folderName):
    if not checkFolderExist(rootPath+"/"+folderName):
        os.mkdir(rootPath+"/"+folderName)
        return True
    else:
        return "Folder already exist"

def formatDate(date, fm=date_format):
    return date.strftime(fm)

def convertToDateObj(date, fm=date_format):
    return datetime.datetime.strptime(date, date_format)

def moveDate(date, dayDelta=0, hourDelta=0):
    if type(date) == str:
        return datetime.datetime.strptime(date, date_format) + datetime.timedelta(days=dayDelta)
    else:
        return date + datetime.timedelta(days=dayDelta)

def PreviousBusinessDay(date, businessDateList):
    if type(date) == str:
        myDate = datetime.datetime.strptime(date, date_format)
    else:
        myDate = date

    while formatDate(myDate) not in businessDateList:
        myDate = moveDate(formatDate(myDate), -1)

    return formatDate(myDate)

def convertTimestampsToDateStr(timestamp):
    s = timestamp[0:10]
    return s

def normalize_headline(row):
    result = row.lower()
    #Delete useless character strings
    result = result.replace('...', ' ')
    whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')
    result = ''.join(filter(whitelist.__contains__, result))
    #result2 = []
    #for c in result:
    #    if c
    return result

def removeStoppingWords(sent):
    result = []
    for w in sent.split(" "):
        if w not in stoppingWordSet:
            result.append(w)
            result.append(" ")
    return "".join(result)

def removeHeading(sent):
    # remove "BRIEF-"
    # remove "BUZZ -"
    # remove "REFILE-BRIEF-"
    # remove "UPDATE "
    # remove "EXCLUSIVE-"
    # remove "Reuters Insider - "
    # remove "BREAKINGVIEWS-"

    headingList = ["BRIEF-", "BUZZ -", "BUZZ-", "REFILE-", "REFILE-BRIEF-", "UPDATE ", "EXCLUSIVE-", "Reuters Insider - ", "BREAKINGVIEWS-"]
    result = sent.lower()
    for h in headingList:
        if h.lower() in result:
            result = result.replace(h.lower(), "")
    return result

def removeOthers(sent):
    wordList = ["holding", "holdings", "ltd"]
    result = sent
    for w in wordList:
        if w in result:
            result = result.replace(w, "")
    return result

def precision(y_true, y_pred):
    total = 0
    valid = 0
    for i,j in zip(y_true, y_pred):
        if j == 1:
            total+=1
            if i==1:
                valid+=1
    if total == 0:
        return -1
    else:
        return valid / total

def iaGetTimeSeries(undlName, field, dateFrom, dateTo):

    if type(dateFrom) != str: dateFrom = formatDate(dateFrom)
    if type(dateTo) != str: dateTo = formatDate(dateTo)

    df = pd.read_csv(dataRootPathMarketData+undlName.split('.')[0] + '_'+undlName.split('.')[1]+'.csv')
    df = df[df.Date >= dateFrom]
    df = df[df.Date <= dateTo]
    df = df.set_index(["Date"])

    return pd.DataFrame(df[field])

def createUndlDataFrame(undlName, undlNameFullNameList, newsSource, filterFuncList, dateFrom, dateTo,
                       benchmark = ""):

    print("Loading", undlName, dateFrom, dateTo, end=" ")
    # get news headlines
    df_list = []
    dateRef = datetime.datetime.strptime(dateFrom, date_format)
    while dateRef <= datetime.datetime.strptime(dateTo, date_format):
        df_list.append(pd.read_csv(dataRootPathNews + formatDate(dateRef) + "/" + undlName + "_headlines.csv"))
        dateRef = moveDate(dateRef, 1)
    news_df = pd.concat(df_list, axis=0)

    # rename and sort columns
    cols = news_df.columns
    news_df.columns = ["timestamp"] + list(cols[1:])
    news_df = news_df.sort_values(["timestamp"])
    news_df.loc[:,"date"] = news_df["versionCreated"].apply(convertTimestampsToDateStr)

    # return empty df if no data
    if news_df.shape[0] == 0:
        print("    done")
        return pd.DataFrame({"date": [], "undlName":[], "sourceCode": [], "storyId":[], "text": [], "oneDayReturn": [], "twoDayReturn": [], "threeDayReturn": []})

    # get market data
    start = min(news_df.date)
    end = max(news_df.date)
    spot_df = iaGetTimeSeries(undlName, "CLOSE", moveDate(start, -10), moveDate(end, 10))
    if benchmark != "":
        spot_df_benchmark = iaGetTimeSeries(benchmark, "CLOSE", moveDate(start, -10), moveDate(end, 10))
        spot_df_benchmark = spot_df_benchmark.loc[spot_df.index]

    # truncate news_df when stock has limited historical data
    news_df = news_df[(news_df.date >= min(spot_df.index))]

    # create one day, two day and three day change columns
    if benchmark != "":
        spot_df.loc[:,"Future-1"] = spot_df.CLOSE.shift(-1)
        spot_df.loc[:,"Future-2"] = spot_df.CLOSE.shift(-2)
        spot_df.loc[:,"Future-3"] = spot_df.CLOSE.shift(-3)
        spot_df = spot_df.iloc[:-3,]
        spot_df_benchmark.loc[:,"Future-1"] = spot_df_benchmark.CLOSE.shift(-1)
        spot_df_benchmark.loc[:,"Future-2"] = spot_df_benchmark.CLOSE.shift(-2)
        spot_df_benchmark.loc[:,"Future-3"] = spot_df_benchmark.CLOSE.shift(-3)
        spot_df_benchmark = spot_df_benchmark.iloc[:-3,]

        spot_df.loc[:,"oneDayReturn"] = \
        np.log(spot_df["Future-1"].values / spot_df["CLOSE"].values)-np.log(spot_df_benchmark["Future-1"].values / spot_df_benchmark["CLOSE"].values)

        spot_df.loc[:,"twoDayReturn"] = \
        np.log(spot_df["Future-2"].values / spot_df["CLOSE"].values)-np.log(spot_df_benchmark["Future-2"].values / spot_df_benchmark["CLOSE"].values)

        spot_df.loc[:,"threeDayReturn"] = \
        np.log(spot_df["Future-3"].values / spot_df["CLOSE"].values)-np.log(spot_df_benchmark["Future-3"].values / spot_df_benchmark["CLOSE"].values)
    else:
        spot_df.loc[:,"Future-1"] = spot_df.CLOSE.shift(-1)
        spot_df.loc[:,"Future-2"] = spot_df.CLOSE.shift(-2)
        spot_df.loc[:,"Future-3"] = spot_df.CLOSE.shift(-3)
        spot_df = spot_df.iloc[:-3,]

        spot_df.loc[:,"oneDayReturn"] = np.log(spot_df["Future-1"].values / spot_df["CLOSE"].values)
        spot_df.loc[:,"twoDayReturn"] = np.log(spot_df["Future-2"].values / spot_df["CLOSE"].values)
        spot_df.loc[:,"threeDayReturn"] = np.log(spot_df["Future-3"].values / spot_df["CLOSE"].values)

    oneDayReturnDict = {d:v for d,v in zip(spot_df.index, spot_df["oneDayReturn"])}
    twoDayReturnDict = {d:v for d,v in zip(spot_df.index, spot_df["twoDayReturn"])}
    threeDayReturnDict = {d:v for d,v in zip(spot_df.index, spot_df["threeDayReturn"])}

    # create concat df, news and log-chg
    businessDateList = list(spot_df.index)
    d = news_df.date.values
    oneDay = []
    twoDay = []
    threeDay = []
    for i in range(len(news_df)):
        oneDay.append(oneDayReturnDict[PreviousBusinessDay(d[i], businessDateList)])
        twoDay.append(twoDayReturnDict[PreviousBusinessDay(d[i], businessDateList)])
        threeDay.append(threeDayReturnDict[PreviousBusinessDay(d[i], businessDateList)])

    news_df.loc[:,"oneDayReturn"] = oneDay
    news_df.loc[:,"twoDayReturn"] = twoDay
    news_df.loc[:,"threeDayReturn"] = threeDay

    # data preprocessing
    fil_df = news_df[news_df["sourceCode"]==newsSource]
    fil_df.loc[:,"text"] = fil_df.text.apply(lambda x: x.lower())

    for f in filterFuncList:
        fil_df.loc[:,"text"] = fil_df.text.apply(f).values
    tmp = []
    for name in undlNameFullNameList:
        tmp.append(fil_df[fil_df.text.apply(lambda x: name in x)])
    fil_df = pd.concat(tmp, axis=0)

    if fil_df.shape[0] == 0:
        df = pd.DataFrame({"date": [], "undlName":[], "sourceCode": [], "storyId":[], "text": [], "oneDayReturn": [], "twoDayReturn": [], "threeDayReturn": []})
    else:
        fil_df["undlName"] = [undlName for i in range(len(fil_df))]
        df = fil_df[["date", "undlName", "sourceCode", "storyId", "text", "oneDayReturn", "twoDayReturn", "threeDayReturn"]]
    print("    done")
    return df

def elmo_vector(x):

    if type(x) == list:
        embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
    else:
        embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tf.reduce_mean(embeddings, 1))

def build_model():

    input_layer = keras.layers.Input(shape=(1024,))

    h = keras.layers.Dropout(rate=0.2)(input_layer)

    prediction = keras.layers.Dense(1, activation="sigmoid")(h)

    model = keras.Model(inputs=[input_layer], outputs=prediction)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Precision()])

    return model

def createNewsHeadlinePrediction(ex, sector_list):

    # remove hub folder
    shutil.rmtree(tf_hub_path)

    undlNameList = []
    for sector in sector_list:
        undlNameList += getUndlNameList(sector)

    start_date = formatDate(today)
    end_date = formatDate(today)

    resultDict = {"undlName":[], "bull_signals":[], "bear_signals":[]}

    # load model
    market_bull_model = build_model()
    market_bull_model.reset_states()
    market_bull_model.load_weights(modelPath + ex + "_market_bull_model.h5")
    market_bear_model = build_model()
    market_bear_model.reset_states()
    market_bear_model.load_weights(modelPath + ex + "_market_bear_model.h5")

    sectorBullModelDict = {}
    sectorBearModelDict = {}
    for sector in sector_list:
        model = build_model()
        model.reset_states()
        model.load_weights(modelPath + sector + "_bull_model.h5")
        sectorBullModelDict[sector] = model

        model = build_model()
        model.reset_states()
        model.load_weights(modelPath + sector + "_bear_model.h5")
        sectorBearModelDict[sector] = model

    tmp = []
    for undlName in undlNameList:
        tmp_df = createUndlDataFrame(undlName, undlNameFullNameDict[undlName], "NS:RTRS",
                                    [removeHeading, normalize_headline, removeOthers],
                                    start_date, end_date, "")
        tmp_df = tmp_df.drop_duplicates(subset='storyId')
        tmp_df = tmp_df.sort_values(["date"])
        if len(tmp_df) != 0: tmp.append(tmp_df)

    if len(tmp) != 0:
        df = pd.concat(tmp, axis=0)
    else:
        print("No News Headlines")
        return True

    print(df.shape)

    # create ELMo Vector
    batch = [df["text"].values[i:i+100] for i in range(0, df.shape[0], 100)]
    batch_elmo = [elmo_vector(x) for x in batch]
    elmo_vector_list = np.concatenate(batch_elmo, axis=0)

    market_bull_model_result = market_bull_model.predict(elmo_vector_list).reshape(-1)
    market_bear_model_result = market_bear_model.predict(elmo_vector_list).reshape(-1)

    sector_bull_model_result = []
    sector_bear_model_result = []
    i = 0
    for undlName in df["undlName"].values:
        sector_bull_model = sectorBullModelDict[getSector(undlName)]
        sector_bear_model = sectorBearModelDict[getSector(undlName)]

        sector_bull_model_result += list(sector_bull_model.predict(elmo_vector_list[i].reshape(1, -1)).reshape(-1))
        sector_bear_model_result += list(sector_bear_model.predict(elmo_vector_list[i].reshape(1, -1)).reshape(-1))
        i += 1

    sector_bull_model_result = np.array(sector_bull_model_result)
    sector_bear_model_result = np.array(sector_bear_model_result)

    resultDict["undlName"] += list(df["undlName"].values)
    resultDict["bull_signals"] += [1 if i > 1 else 0 for i in market_bull_model_result + sector_bull_model_result]
    resultDict["bear_signals"] += [1 if i > 1 else 0 for i in market_bear_model_result + sector_bear_model_result]

    result_df = pd.DataFrame.from_dict(resultDict)
    to_drop = [i for i in range(result_df.shape[0]) if result_df.iloc[i, 1] == 0 and result_df.iloc[i, 2] == 0]
    result_df = result_df.drop(to_drop)
    result_df.to_csv(r"D:/python/EventDriven/result/" + formatDate(today) + "_" + ex + ".csv")

    return True

def main():

    sector_list = ["Tencent", "Chinese_Bank", "Chinese_Insurance", "Chinese_Oil", "Chinese_Auto",
               "Chinese_Telecom", "Chinese_Industrial", "HK_Property", "HK_Bank"]

    createNewsHeadlinePrediction(ex="HK", sector_list=sector_list)

if __name__=="__main__":
    main()
