import robin_stocks as r
import json
from datetime import date

# *****NOTE: won't be able to run without two factor login message sent to my phone*******

# use this api to request stock names and ticker names for top movers of a day
# check https://robin-stocks.readthedocs.io/en/latest/ for more information about this API

# use the robin_stocks library to login to my account
login = r.login('scelmore1@gmail.com', '!@Sce21893')

# get the stocks that have moved most up or down over the day
stocks_up = r.get_top_movers('up')
stocks_down = r.get_top_movers('down')

# create a 'portfolio' dictionary of these stocks
portfolio = []
for stock in stocks_up:
    portfolio.append({'Symbol': stock['symbol'],
                      'Name': r.get_name_by_symbol(stock['symbol'])})

for stock in stocks_down:
    portfolio.append({'Symbol': stock['symbol'],
                      'Name': r.get_name_by_symbol(stock['symbol'])})

portfolio_dict = {str(date.today()): portfolio}

# write the stocks to a json file, update if adding on an extra day
with open('stock_portfolio.json', 'r') as json_file:
    dict_data = json.load(json_file)
    portfolio_dict.update(dict_data)

with open('stock_portfolio.json', 'w') as json_file:
    json.dump(portfolio_dict, json_file)
