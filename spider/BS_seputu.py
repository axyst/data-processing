import requests
import chardet
import re
import csv
from bs4 import BeautifulSoup
r = requests.get('http://seputu.com')
r.encoding = chardet.detect(r.content)['encoding']
soup = BeautifulSoup(r.text, 'html.parser', from_encoding='utf-8')
rows = []
for mulu in soup.find_all(class_='mulu'):
    h2 = mulu.find('h2')
    if h2 is not None:
        h2_title = h2.string
        print(h2_title)
        for a in mulu.find(class_='box').find_all('a'):
            href = a.get('href')
            box_title = a.get('title')
            pattern = re.compile(r'\s*\[(.*)\]\s+(.*)')
            match = pattern.search(box_title)
            if match is not None:
                date = match.group(1).encode('utf-8')
                real_title = match.group(2)
                content = (h2_title, real_title, href, date)
                rows.append(content)
                print(h2_title, real_title, href, date)
headers = ['chapter', 'title', 'href', 'date']
with open('seputu.csv', 'w') as f:
    f_csv = csv.writer(f,)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
