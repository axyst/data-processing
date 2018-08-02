# coding:utf-8

import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import chardet
import requests
import os


def Schedule(blocknum, blocksize, totalsize):
    per = 100.0 * blocknum * blocksize / totalsize
    if per > 100:
        per = 100
    print('当前下载进度：%d' % per)


user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent': user_agent}
r = requests.get(
    'http://www.ivsky.com/tupian/ziranfengguang/',
    headers=headers)
r.encoding = chardet.detect(r.content)['encoding']
soup = BeautifulSoup(r.text, 'html.parser', from_encoding='utf-8')
i = 1
for a in soup.find_all('img', src=True):
    img_url = a['src']
    print(img_url)
    dir = os.path.abspath('./saved/')
    work_path = os.path.join(dir, 'img' + str(i) + '.jpg')
    urllib.request.urlretrieve(img_url, work_path, Schedule)
    print(f'image {i} completed')
    i = i + 1
