from googleapiclient.discovery import build
import http.client
import urllib.request
from urllib.request import Request, urlopen
from time import sleep

def from_google():
    c_x = "009171367015975864750:tzgp6vvag9k"
    f=open("C:/Users/bzlis/api_keys.txt", "r")
    clarifai_key = f.readline().strip()
    developer_key = f.readline().strip()
    f.close()
    service = build("customsearch", "v1",developerKey=developer_key)

    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    pg = 1
    count = 0
    while (pg < 100):
        for i in range(0, 10):
            res = service.cse().list(q="infographic", cx=c_x,  num=10, searchType="image", start=pg).execute()
            retUrl = res['items'][i]['link']
            extension = "." + retUrl.split("/")[-1].split(".")[-1]
            extension = extension[:4]
            urllib.request.urlretrieve(retUrl, "images/ig" + str(count) + extension)
            count += 1
        pg += 10
        sleep(0.03)
