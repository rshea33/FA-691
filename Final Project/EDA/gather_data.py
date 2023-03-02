import bs4 as bs
import requests
import yfinance as yf
import datetime

def get_tickers(): # Kindly borrowed from: https://wire.insiderfinance.io/how-to-get-all-stocks-from-the-s-p500-in-python-fbe5f9cb2b61

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    tickers = [s.replace('\n', '') for s in tickers]

    return tickers





def main():
    tickers = get_tickers()
    print("tickers = ", tickers, sep = '\n')
    start = datetime.datetime(2019, 1, 1)
    stop = datetime.datetime(2022, 1, 31)

    df = yf.download(tickers, start, stop)['Adj Close']
    print("data.shape = ", df.shape)
    df.to_csv('data.csv')

if __name__ == '__main__':
    main()

