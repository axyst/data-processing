# coding:utf-8
import requests


class HTMLDownloader(object):

    def download(self, url):
        if url is None:
            return None
        headers = {
            'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            r.encoding = 'utf-8'
            return r.text
        return None
