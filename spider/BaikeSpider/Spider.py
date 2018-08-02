from DataOutput import DataOutput
from HTMLDownloader import HTMLDownloader
from URLManager import URLManager
from HTMLParser import HTMLParser


class Spider(object):
    def __init__(self):
        self.manager = URLManager()
        self.downloader = HTMLDownloader()
        self.parser = HTMLParser()
        self.output = DataOutput()

    def crawl(self, root_url, crawl_size):
        self.manager.add_new_url(root_url)
        while self.manager.has_new_url() and self.manager.old_url_size() < crawl_size:
            try:
                new_url = self.manager.get_new_url()
                html = self.downloader.download(new_url)
                new_urls, data = self.parser.parser(new_url, html)
                self.manager.add_new_urls(new_urls)
                self.output.store_data(data)
                print('%s links crawled' % self.manager.old_url_size())
            except Exception:
                print('crawl failed')
        self.output.output_html()


if __name__ == '__main__':
    spider = Spider()
    crawl_size = int(input('Please input crawl number'))
    init_page = input('Please input first page')
    spider.crawl(
        init_page, crawl_size)
