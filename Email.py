import numpy as np
import pandas as pd
import os
import win32com.client as win32
import datetime

path = r"D:/python/EventDriven/result/"
date_format = "%Y-%m-%d"

def formatDate(date, fm=date_format):
    return date.strftime(fm)

def moveDate(date, dayDelta=0, hourDelta=0):
    if type(date) == str:
        return datetime.datetime.strptime(date, date_format) + datetime.timedelta(days=dayDelta) + datetime.timedelta(hours=hourDelta)
    else:
        return date + datetime.timedelta(days=dayDelta) + + datetime.timedelta(hours=hourDelta)

def email(to, sub, HTMLBody, attachmentURLList):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = to
    mail.Subject = sub
    mail.HTMLBody = HTMLBody
    for url in attachmentURLList:
        if os.path.exists(url): mail.Attachments.Add(url)
    mail.Send()

def main():
    today = datetime.datetime.now()
    url1 = path + formatDate(today) + "_HK.csv"
    url2 = path + formatDate(today) + "_AX.csv"
    url3 = path + formatDate(today) + "_SI.csv"
    email("isaac.law@rbccm.com", "Eikon_News", "", [url1, url2, url3])

if __name__=="__main__":
    main()
