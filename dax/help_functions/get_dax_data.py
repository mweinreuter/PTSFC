import yfinance as yf


def get_data():

    msft = yf.Ticker("^GDAXI")
    daxdata = msft.history(period="max")

    return (daxdata)
