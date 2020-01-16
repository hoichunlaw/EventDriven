import eikon as ek
import numpy as np
import pandas as pd
import os
import zipfile
import datetime
import cufflinks as cf
import configparser as cp

ek.set_app_key('e4ae85e1e08b47ceaa1ee066af96cabe6e56562a')

dataRootPath = r"D:/Eikon_Data/"
dataRootPathNews = r"D:/Eikon_Data/News/"
dataRootPathMarketData = r"D:/Eikon_Data/Market_Data/"
databasePath = r"D:/Database/"
zipFolderPath = r"D:/Zip_Folder/"
date_format = "%Y-%m-%d"

def checkFolderExist(path):
    return os.path.isdir(path)

def checkFileExist(path):
    return os.path.isfile(path)

def createFolder(rootPath, folderName):
    if rootPath[-1] == "/":
        myRootPath = rootPath[:-1]
    else:
        myRootPath = rootPath
    if not checkFolderExist(myRootPath+"/"+folderName):
        os.mkdir(myRootPath+"/"+folderName)
        return True
    else:
        return "Folder already exist"

def formatDate(date, fm=date_format):
    return date.strftime(fm)

def moveDate(date, dayDelta=0, hourDelta=0):
    if type(date) == str:
        return datetime.datetime.strptime(date, date_format) + datetime.timedelta(days=dayDelta) + datetime.timedelta(hours=hourDelta)
    else:
        return date + datetime.timedelta(days=dayDelta) + + datetime.timedelta(hours=hourDelta)

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def iaZipFolder(path):
    if path[-1] == '/':
        zipFileName = path.split("/")[-2] + "_zip.zip"
    else:
        zipFileName = path.split("/")[-1] + "_zip.zip"

    if checkFileExist(zipFolderPath + zipFileName): os.remove(zipFolderPath + zipFileName)
    zipf = zipfile.ZipFile(zipFolderPath + zipFileName, 'w', zipfile.ZIP_DEFLATED)
    zipdir(path, zipf)
    zipf.close()

def downloadNews(undlName, date, savePath):
    if not checkFolderExist(savePath + formatDate(date)):
        createFolder(savePath, formatDate(date))

    # download data
    df = ek.get_news_headlines("R:"+undlName+" and english",
                               date_from=formatDate(moveDate(date,-1)) + "T16:00:00",
                               date_to=formatDate(moveDate(date)) + "T16:00:00",
                               count=100)

    # move date back to HK time
    df.index = moveDate(np.array(list(df.index)),0,8)
    df.versionCreated = moveDate(np.array(list(df.versionCreated)),0,8)
    # save data
    df.to_csv(savePath + formatDate(date) + "/" + undlName + "_headlines.csv")

def downloadHistoricalNews(undlName, dateFrom, dateTo, savePath):
    if type(dateFrom) == str:
        myDateFrom = datetime.datetime.strptime(dateFrom, date_format)
    else:
        myDateFrom = dateFrom

    if type(dateTo) == str:
        myDateTo = datetime.datetime.strptime(dateTo, date_format)
    else:
        myDateTo = dateTo

    dateRef = myDateFrom
    while dateRef <= myDateTo:
        print("Download", undlName, dateRef)
        downloadNews(undlName, dateRef, savePath)
        dateRef = moveDate(dateRef, 1)

def downloadMarketData(undlName, date, savePath):

    # download data
    try:
        df_new = ek.get_timeseries(undlName, fields=["CLOSE", "HIGH", "LOW", "OPEN", "VOLUME"],
                                   start_date=formatDate(date), end_date=formatDate(date), interval="daily", corax="adjusted")
    except:
        df_new = []

    if type(df_new) == pd.core.frame.DataFrame:
        myUndlName = undlName.split('.')[0] + '_' + undlName.split('.')[1]
        df_new.index = pd.Series(df_new.index).apply(formatDate)
        if checkFileExist(savePath + myUndlName + ".csv"):
            df = pd.read_csv(savePath + myUndlName + ".csv")
            df = df.set_index("Date")
            if df_new.index[0] not in list(df.index):
                df = pd.concat([df, df_new], axis=0)
                df.to_csv(savePath + myUndlName + ".csv")
        else:
            df_new.to_csv(savePath + myUndlName + ".csv")

def downloadHistoricalMarketData(undlName, dateFrom, dateTo, savePath):

    # download data
    df = ek.get_timeseries(undlName, fields=["CLOSE", "HIGH", "LOW", "OPEN", "VOLUME"],
                           start_date=dateFrom, end_date=dateTo, interval="daily", corax="adjusted")
    df.index = pd.Series(df.index).apply(formatDate)
    myUndlName = undlName.split('.')[0] + '_' + undlName.split('.')[1]
    df.to_csv(savePath + myUndlName + ".csv")

def main():
    today = datetime.datetime.now()
    df = pd.read_csv(r'D:/Database/Underlying_Database/undlNameList.csv')
    undlNameList = list(df.undlName.values)

    # download News Headlines
    for undlName in undlNameList:
        print("Download", undlName, today)
        downloadNews(undlName, today, dataRootPathNews)

if __name__=="__main__":
    main()
