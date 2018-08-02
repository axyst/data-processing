# coding:utf-8
import codecs


class DataOutput(object):
    def __init__(self):
        self.datas = []

    def store_data(self, data):
        if data is None:
            return
        self.datas.append(data)

    def output_html(self):
        fout = codecs.open('baike.html', 'w', encoding='utf-8')
        fout.write('<html>')
        fout.write('<body>')
        fout.write('<table>')
        fout.write('<tr>')
        fout.write('<td>序号</td>')
        fout.write('<td>URL</td>')
        fout.write('<td>标题</td>')
        fout.write('<td>简介</td>')
        fout.write('</tr>')
        fout.close()
        fout = codecs.open('baike.html', 'a', encoding='utf-8')
        cnt = 0
        total = 1
        for data in self.datas:
            fout.write('<tr>')
            fout.write(f'<td>{total}</td>')
            fout.write(f'<td>{data["url"]}</td>')
            fout.write(f'<td>{data["title"]}</td>')
            fout.write(f'<td>{data["summary"]}</td>')
            fout.write('</tr>')
            total = total + 1
            if cnt < 100:
                cnt = cnt + 1
            else:
                fout.close()
                fout = codecs.open('baike.html', 'a', encoding='utf-8')
                cnt = 0
        fout.close()
        fout = codecs.open('baike.html', 'a', encoding='utf-8')
        fout.write('</html>')
        fout.write('</body>')
        fout.write('</table>')
        fout.close()
