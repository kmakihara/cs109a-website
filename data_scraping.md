---
title: Replace states with their abbreviations
notebook: data_scraping.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}



```python
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import csv
from os import listdir
from os.path import isfile, join
%matplotlib inline
```


## 1: Scrape Data from XLS, Clean CSV's

### 1.1 Clean FBI and Census Data



```python
fbi_data_path = 'Crime/data/fbi_data/'
```




```python
def xls_to_csv(wb, csv_name):
    wb = xlrd.open_workbook(wb)
    sh = wb.sheet_by_index(0)
    output_file = open('{}.csv'.format(csv_name), 'w')
    wr = csv.writer(output_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    output_file.close()
```




```python
def get_crime_stats(df_path):
    df = pd.read_csv(df_path, skiprows=3)
    df['Metropolitan Statistical Area'] = df['Metropolitan Statistical Area'].fillna(method='ffill')
    crime_stats = df[df['Counties/principal cities'] == 'Total area actually reporting']
    crime_stats = crime_stats.drop(['Counties/principal cities', 'Population'], axis=1)
    columns = list(map(lambda x: '_'.join(x.split()), crime_stats.columns))
    crime_stats.columns = columns
    crime_stats['Metropolitan_Statistical_Area'] = crime_stats['Metropolitan_Statistical_Area'].apply(lambda x: x.split('M.S.A')[0].strip())
    mask = crime_stats['Metropolitan_Statistical_Area'].apply(lambda x: 'M.D' not in x)
    crime_stats = crime_stats[mask]
    crime_stats = pd.DataFrame(crime_stats[['Metropolitan_Statistical_Area', 'Murder_and_nonnegligent_manslaughter']])
    return crime_stats
```




```python
fbi_filepaths = [f for f in listdir(fbi_data_path) if isfile(join(fbi_data_path, f))]
```




```python
for f in fbi_filepaths:
    file_path = join(fbi_data_path, f)
    output_path = 'csvs/{}'.format(f[:-4])
    xls_to_csv(file_path, output_path)
```




```python
csv_filepaths = [f for f in listdir('csvs/') if isfile(join('csvs/', f))]
```




```python
df = pd.read_csv(join('csvs/', csv_filepaths[0]), skiprows=3)
```




```python
for f in csv_filepaths:
    file_path = join('csvs/', f)
    df = get_crime_stats(file_path)
    for col in df.columns:
        if 'Unnamed' in col:
            df = df.drop(col, axis=1)
    df.to_csv('csvs/{}.csv'.format(f[:-4]), index=False)
```


### Description of Data

Both datasets are continuous, with counts in certain buckets. The data is not extremely difficult to deal with, though it does require some cleaning and conversion. Data from the FBI database is downloaded in XLS format, which is then parsed into a CSV. The CSV is then cleaned to only keep relevant bits of information, specifically the murder counts/

The Census data is downloaded in a big CSV file form. This information is cleaner and more standardized, since it is already in csv form. However, there are still issues of missing values here and there. When combining each dataframe into a single joint one, we do an inner join on the MSA's present in each dataset. This is because each data set is only as useful as the other data set. This also means we need to be extra stringent when it comes to parsing the MSA names, so we don't lose too much data. Whenever there is a NaN value, we either drop it or set it to 0, depending on the situation. If it is a percentage category, then we set it to 0, since there are likely other categories with positive percentages.

This data is cleaned, but only in a cursory fashion to perform EDA. As we venture into training our model, we will look into some more one-hot encoding, as well as potential classification measures (judging an area as low risk, medium risk, high risk etc...).

### 1.2 Clean Additional Data, ATF reports on gun recoveries and sources by state



```python
atf_data_path = 'Crime/data/atf_data/'
```




```python
atf_filepaths = [f for f in listdir(atf_data_path) if isfile(join(atf_data_path, f))]
```




```python
for f in atf_filepaths:
    file_path = join(atf_data_path, f)
    output_path = 'atf_csvs/{}'.format(f[-9:-5])
    xls_to_csv(file_path, output_path)
```




```python
atf_csvs = [f for f in listdir('atf_csvs') if isfile(join('atf_csvs', f))]
```




```python
frames = []
for f in atf_csvs:
    file_path = join('atf_csvs', f)
    output_path = 'atf_csvs/{}'.format(f[:-4])
    df = pd.read_csv(file_path, skiprows=1)
    col_one = df.columns[0]
    col_two = df.columns[1]
    df = df.drop(col_one, axis=1)
    df = df.rename(index=str, columns={col_two: 'Source State'})
    df = df.dropna()
    frames.append(df.dropna())
```




```python
states = [
            ['Arizona', 'AZ'],
            ['Alabama', 'AL'],
            ['Alaska', 'AK'],
            ['Arkansas', 'AR'],
            ['California', 'CA'],
            ['Colorado', 'CO'],
            ['Connecticut', 'CT'],
            ['Delaware', 'DE'],
            ['Florida', 'FL'],
            ['Georgia', 'GA'],
            ['Hawaii', 'HI'],
            ['Idaho', 'ID'],
            ['Illinois', 'IL'],
            ['Indiana', 'IN'],
            ['Iowa', 'IA'],
            ['Kansas', 'KS'],
            ['Kentucky', 'KY'],
            ['Louisiana', 'LA'],
            ['Maine', 'ME'],
            ['Maryland', 'MD'],
            ['Massachusetts', 'MA'],
            ['Michigan', 'MI'],
            ['Minnesota', 'MN'],
            ['Mississippi', 'MS'],
            ['Missouri', 'MO'],
            ['Montana', 'MT'],
            ['Nebraska', 'NE'],
            ['Nevada', 'NV'],
            ['New Hampshire', 'NH'],
            ['New Jersey', 'NJ'],
            ['New Mexico', 'NM'],
            ['New York', 'NY'],
            ['North Carolina', 'NC'],
            ['North Dakota', 'ND'],
            ['Ohio', 'OH'],
            ['Oklahoma', 'OK'],
            ['Oregon', 'OR'],
            ['Pennsylvania', 'PA'],
            ['Rhode Island', 'RI'],
            ['South Carolina', 'SC'],
            ['South Dakota', 'SD'],
            ['Tennessee', 'TN'],
            ['Texas', 'TX'],
            ['Utah', 'UT'],
            ['Vermont', 'VT'],
            ['Virginia', 'VA'],
            ['Washington', 'WA'],
            ['West Virginia', 'WV'],
            ['Wisconsin', 'WI'],
            ['Wyoming', 'WY'],
        ]

states = list(map(lambda x: [x[0].lower(), x[1]], states))
```




```python
states.append(['district of columbia', 'DC'])
```




```python
def abbreviate(state_name, state_list=states):
    lowercase = state_name.lower()
    abbreviation = [x[1] for x in state_list if x[0] == lowercase]
    if len(abbreviation) > 0:
        return abbreviation[0]
    else:
        return state_name
```




```python
cleaned_frames = []
for i, frame in enumerate(frames):
    df_left = pd.DataFrame(frame['Source State'])
    source_totals = frame.columns[-1]
    df_left['Total Sourced'] = frame[source_totals]
    recovery_totals = frame['Source State'][-1]
    df_right = frame[frame['Source State'] == recovery_totals].T
    df = pd.merge(df_left, df_right, left_on='Source State', right_index=True)
    cols = ['State', 'Total Sourced', 'Total Recovered']
    df.columns = cols
    df['State'] = df['State'].apply(abbreviate, state_list=states)
    cleaned_frames.append(df)
    df.to_csv(join('atf_csvs', atf_csvs[i]), index=False)
```


Here I recover and clean data from the ATF database. There are 3 columns in this df: state, total sourced, and total recovered. Total sourced is the number of firearms the ATF recovered in that year that were sourced from that state. Similarly, the total recovered is the number of firearms the ATF recovered in that state. I also abbreviated each state because that is the format in the MSA analysis.



```python

```
