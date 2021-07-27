'''
We aim to analyse the London Smart Meter data set. 

The aim is to conduct cleaning so as to make it available for other uses and also to
conduct some basic analysis. 

'''

from numpy.core.shape_base import block
import pandas as pd
import os
import matplotlib.pyplot as plt
from os import walk
import datetime



os.getcwd()
os.chdir('/home/usman/Documents/Smart-Meter-analysis')

## total number of files
dir = '/home/usman/Documents/Smart-Meter-analysis/archive/halfhourly_dataset/halfhourly_dataset'
list = os.listdir(dir) # dir is your directory path
number_files = len(list)

## list of filenames
_, _, filenames = next(walk(dir))

# load all files 
#df = pd.concat([pd.read_csv(dir + '/' + i) for i in filenames])

#load single file
df = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/archive/halfhourly_dataset/halfhourly_dataset/block_0.csv')

df.head(25)

# write function to datetime

df.dtypes
df['tstp']  = pd.to_datetime(df['tstp'], format='%Y-%m-%d %H:%M:%S.%f')
df['Date']  =  pd.to_datetime(df['tstp'], format='%Y-%M-%D').dt.date

df.columns = ['LCLid','tstp','energy','Date']

min_max_dates = df.groupby('LCLid')['Date'].agg(Min = 'min', Max = 'max',)
check_na = df.energy.isnull().groupby(df['LCLid']).sum()

'''
If we loop through the filenames, we want to:
1. load each file
2. convert timestamp to datetime
3. For each ID, assess when the min/max date is and then calculate the difference
4. For each ID, assess missing data/NA 
5. for each ID, check how many entries we should have based on diff calculated above,
and see how many we actually have 
'''


def blockwise_analysis(x,dir):
    df_analysis = pd.DataFrame()
    df = pd.read_csv(dir + '/' + x)
    df['tstp']  = pd.to_datetime(df['tstp'], format='%Y-%m-%d %H:%M:%S.%f')
    df['Date']  =  pd.to_datetime(df['tstp'], format='%Y-%M-%D').dt.date

    df.columns = ['LCLid','tstp','energy','Date']

    min_max_dates = df.groupby('LCLid')['Date'].agg(Min = 'min', Max = 'max')
    check_na = df.energy.isnull().groupby(df['LCLid']).sum()
    new_df = min_max_dates.merge(check_na, how='inner', on='LCLid')
    df_analysis = pd.concat([df_analysis,new_df])
    return df_analysis
  



df_analysis = [blockwise_analysis(x,dir) for x in filenames]
df_ = pd.concat(df_analysis)

## check the number of unique ID
df_.index.unique
## 5600 unique id's

## check the number of na's 
sum(df_['energy'] > 0)

## we need to calc the difference in dates 

df_[['Min','Max']] = df_[['Min','Max']].apply(pd.to_datetime) #if conversion required
df_['diff'] = (df_['Max'] - df_['Min']).dt.days

plt.hist(df_['diff'])
plt.show()


df_ids = df.LCLid.unique()

dfs_av = {}
for x in df_ids:
    check_df = df_.filter(like = x, axis=0)
    dfs_av[x] = check_df.shape[0]

df_test = df[df['LCLid'] == 'MAC000002' ]


min_date = df_test['Date'].min()
max_date = df_test['Date'].max()

date_set = pd.date_range(min_date, max_date)
diff_date = max_date - min_date

df_test = df_test.set_index('Date')
df_test = df_test.reindex(date_set, fill_value=1000.00)
# One of the things that need to be looked at is if the data is sequential. 



# we also want to calculate either day or weekly profiles.
# on top of these, we then want to conduct other analysis, i.e. feature extraction and 
# clustering etc


'''
1. load each LCLid
2. create df with weekly timeseries, i.e. start monday midnight  and finish after 7 days
    a. find first monday and last sunday of the dataset
    i.e. Date of week start as columns, then following columns as times of that week
3. check for NA's and fill
    a. Fill NA's according time day of the week and time (avg)
    b. save this df as well
4. extract features from week
    a. Min, Max, mean, SD, extract features related to lag/shift etc
5. collect all statistics in one file and save them. 
6. Use this for clustering methods. 
'''

df = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/archive/hhblock_dataset/hhblock_dataset/block_0.csv')
# given the way hhblock dataset is structured, we can easily do single week analysis
test_id = df[df.LCLid == 'MAC000002' ]

# for each week, get the average value for each timestep 

test_id['day'] = pd.to_datetime(test_id['day'], format='%Y-%M-%d')

# we need to get days of the week and then groupby

test_id['dow'] = test_id['day'].apply(lambda time: time.dayofweek)
'''
monday = 0
tuesday = 1
wed = 2
thur = 3
friday = 4
sat = 5
sun = 6
'''

# we want to groupby each (id) and dow and take the mean 
grpbd_df = test_id.groupby('dow').mean()

# create df of avg weekly vals

block_dir = '/home/usman/Documents/Smart-Meter-analysis/archive/hhblock_dataset/hhblock_dataset'
x = 'block_1.csv'
def avg_weekly_vals(x,dir):
    df = pd.read_csv(dir + '/' + x)

    df['day'] = pd.to_datetime(df['day'], format='%Y-%M-%d')

    df['dow'] = df['day'].apply(lambda time: time.dayofweek)
    grpbd_df = df.groupby(['LCLid','dow']).mean()
    return grpbd_df

_, _, filenames = next(walk(block_dir))

avg_df = [avg_weekly_vals(x,block_dir) for x in filenames]

avg_df = pd.concat(avg_df)
avg_df.to_csv('weekly_avg_df.csv')


#create plots for profiles for each day of the week and save them 

#create plot for weekly profiles 