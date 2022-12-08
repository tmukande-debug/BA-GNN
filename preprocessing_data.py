import sys, os, time
from tqdm import tqdm
import pandas as pd
import csv
data_home = 'run_time/data'

def fx_ts(all_ts):
    max_ts, min_ts = max(all_ts), min(all_ts)
    dt = max_ts - min_ts
    print('min, max, dt')
    print(min_ts, max_ts,dt)
    print('days:', dt / (24 * 3600))

def yc_preprocess():
    path = 'yc'
    # N = 1000000
    N = 0
    all_ts1 = []
    all_ts2 = []
    all_ts = []
    sid2vid_list = {}
    # Session ID,Timestamp,Item ID,Price,Quantity
    # 420374,2014-04-06T18:44:58.314Z,214537888,12462,1
    cnt = 0
    pbar = tqdm(desc='read buys')
    with open(f'{data_home}/{path}/yoochoose-buys.dat', 'r') as f:
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))

            all_ts1.append(ts)

            sid2vid_list.setdefault(sid, []) #key-value
            sid2vid_list[sid].append([vid, 0, ts]) # {'SessionID1': [[vid1,0,ts1],…,[vidn,0,tsn]],'Session ID2':…,}
    pbar.close()
    print("------buys:")
    fx_ts(all_ts1)
    print("------buys------")
    # return

    # session_id,timestamp,item_id,category
    # 1,2014-04-07T10:51:09.277Z,214536502,0
    cnt = 0
    pbar = tqdm(desc='read clicks')
    with open(f'{data_home}/{path}/yoochoose-clicks.dat', 'r') as f:
        f.readline()
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')))
            all_ts2.append(ts)

            sid2vid_list.setdefault(sid, [])
            sid2vid_list[sid].append([vid, 1, ts])
    pbar.close()
    print("-----clicks:")
    fx_ts(all_ts2)
    print("-----clicks-----")
    # return

    for sid in sid2vid_list:
        sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

    n = len(sid2vid_list)
    yc = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1])

    frac = 1

    n_part = n // frac
    yc_part = yc[-n_part:]


    out_path = f'{path}_1_{frac}'
    os.mkdir(f'{data_home}/{out_path}')
    with open(f'{data_home}/{out_path}/data.txt', 'w') as f:
        for sid, vid_list in yc_part:
            for i in range(len(vid_list)):
                ts = vid_list[i][-1]
                all_ts.append(ts)
            vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))

            sess = ' '.join([sid, vid_list])
            f.write(sess + '\n')

    print('========data=======')
    fx_ts(all_ts)
    print('========data=======')
    print('session num:', len(yc_part))

    print(yc[-1])

def kaggle_preprocess():
    path = 'kaggle'
    # N = 1000000
    N = 0
    all_ts1 = []
    all_ts2 = []
    all_ts = []
    sid2vid_list = {}
    # Session ID,Timestamp,Item ID,Price,Quantity
    # 420374,2014-04-06T18:44:58.314Z,214537888,12462,1

    #473613801,2019-12-01 00:17:50 UTC,4554,purchase
    cnt = 0
    pbar = tqdm(desc='read buys')
    with open(f'{data_home}/{path}/kaggle-buys.csv', 'r') as f:
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')))

            all_ts1.append(ts)

            sid2vid_list.setdefault(sid, []) #key-value
            sid2vid_list[sid].append([vid, 0, ts]) # {'SessionID1': [[vid1,0,ts1],…,[vidn,0,tsn]],'Session ID2':…,}
    pbar.close()
    print("------buys:")
    fx_ts(all_ts1)
    print("------buys------")
    # return

    # session_id,timestamp,item_id,category
    # 1,2014-04-07T10:51:09.277Z,214536502,0
    #412120092,2019-12-01 00:00:00 UTC,5764655,view
    cnt = 0
    pbar = tqdm(desc='read clicks')
    with open(f'{data_home}/{path}/kaggle-clicks.csv', 'r') as f:
        f.readline()
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')))
            all_ts2.append(ts) #sqy

            sid2vid_list.setdefault(sid, [])
            sid2vid_list[sid].append([vid, 1, ts])
    pbar.close()
    print("-----clicks:")
    fx_ts(all_ts2)
    print("-----clicks-----")
    # return

    for sid in sid2vid_list:
        sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

    n = len(sid2vid_list)
    kg = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1])

    frac = 1

    n_part = n // frac
    kg_part = kg[-n_part:]


    out_path = f'{path}_1_{frac}'
    os.mkdir(f'{data_home}/{out_path}')
    with open(f'{data_home}/{out_path}/data.txt', 'w') as f:
        for sid, vid_list in kg_part:
            for i in range(len(vid_list)):
                ts = vid_list[i][-1]
                all_ts.append(ts)
            vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))

            sess = ' '.join([sid, vid_list])
            f.write(sess + '\n')

    print('========data=======')
    fx_ts(all_ts)
    print('========data=======')
    print('session num:', len(kg_part))

    print(kg[-1])

def kaggledec_preprocess():
    path = 'kaggledec'
    # N = 1000000
    N = 0
    all_ts1 = []
    all_ts2 = []
    all_ts = []
    sid2vid_list = {}
    # Session ID,Timestamp,Item ID,Price,Quantity
    # 420374,2014-04-06T18:44:58.314Z,214537888,12462,1

    #473613801,2019-12-01 00:17:50 UTC,4554,purchase
    cnt = 0
    pbar = tqdm(desc='read buys')
    with open(f'{data_home}/{path}/kaggle-buys.csv', 'r') as f:
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')))

            all_ts1.append(ts)

            sid2vid_list.setdefault(sid, []) #key-value
            sid2vid_list[sid].append([vid, 0, ts]) # {'SessionID1': [[vid1,0,ts1],…,[vidn,0,tsn]],'Session ID2':…,}
    pbar.close()
    print("------buys:")
    fx_ts(all_ts1)
    print("------buys------")
    # return

    # session_id,timestamp,item_id,category
    # 1,2014-04-07T10:51:09.277Z,214536502,0
    #412120092,2019-12-01 00:00:00 UTC,5764655,view
    cnt = 0
    pbar = tqdm(desc='read clicks')
    with open(f'{data_home}/{path}/kaggle-clicks.csv', 'r') as f:
        f.readline()
        for line in f:
            pbar.update(1)
            cnt += 1
            if N > 0 and cnt > N: break
            line = line[:-1]
            sid, ts, vid, _ = line.split(',')
            ts = int(time.mktime(time.strptime(ts[:19], '%Y-%m-%d %H:%M:%S')))
            all_ts2.append(ts)

            sid2vid_list.setdefault(sid, [])
            sid2vid_list[sid].append([vid, 1, ts])
    pbar.close()
    print("-----clicks:")
    fx_ts(all_ts2)
    print("-----clicks-----")
    # return

    for sid in sid2vid_list:
        sid2vid_list[sid] = sorted(sid2vid_list[sid], key=lambda x: x[-1])

    n = len(sid2vid_list)
    kg = sorted(sid2vid_list.items(), key=lambda x: x[1][-1][-1]) #

    frac = 1

    n_part = n // frac
    kg_part = kg[-n_part:]


    out_path = f'{path}_1_{frac}'
    os.mkdir(f'{data_home}/{out_path}')
    with open(f'{data_home}/{out_path}/data.txt', 'w') as f:
        for sid, vid_list in kg_part:
            for i in range(len(vid_list)):
                ts = vid_list[i][-1]
                all_ts.append(ts)
            vid_list = ','.join(map(lambda vid: ':'.join(map(str, vid)), vid_list))

            sess = ' '.join([sid, vid_list])
            f.write(sess + '\n')

    print('========data=======')
    fx_ts(all_ts)
    print('========data=======')
    print('session num:', len(kg_part))

    print(kg[-1])

def main():
    print('START, preprocessing_data.py')
    yc_preprocess()
    #kaggle_preprocess()
    #kaggledec_preprocess()
if __name__ == '__main__':
    main()


