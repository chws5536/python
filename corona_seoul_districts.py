import requests
from datetime import datetime
import pandas as pd

# prepare parameters
key = 'YOUR_API_KEY'
file_type = 'json'
date = datetime.now().strftime('%Y.%m.%d') + '.00'

# get the data
# 서울특별시 공공데이터를 사용한 결과
url = f'http://openapi.seoul.go.kr:8088/{key}/{file_type}/TbCorona19CountStatusJCG/1/5/{date}'
r = requests.get(url).json()
data = r['TbCorona19CountStatusJCG']['row']

# check response
response = r['TbCorona19CountStatusJCG']['RESULT']['MESSAGE']
if response != '정상 처리되었습니다':
    print(response)

# check if already up to date by comparing with previous data
# load previous data
df = pd.read_json('corona_seoul_districts.json')

# compare the lastest date of previous data with the new data's date
if df['JCG_DT'].values[-1] == data[0]['JCG_DT']:
    print('no new update')
    quit()
else:
    print('update available')

# add new data and save
df_new = pd.DataFrame(data)
df_updated = pd.concat([df, df_new], ignore_index=True)
print(df_updated)
df_updated.to_json('corona_seoul_districts.json')


