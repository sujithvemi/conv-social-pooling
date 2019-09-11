import pandas as pd
import numpy as np
import multiprocessing as mp
from scipy.io import savemat
from tqdm import tqdm

us101_1 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0750am-0805am/trajectories-0750am-0805am.txt'
us101_2 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0805am-0820am/trajectories-0805am-0820am.txt'
us101_3 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0820am-0835am/trajectories-0820am-0835am.txt'
i80_1 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0400pm-0415pm/trajectories-0400-0415.txt'
i80_2 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0500pm-0515pm/trajectories-0500-0515.txt'
i80_3 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0515pm-0530pm/trajectories-0515-0530.txt'

# us101_1 = '/home/sujith/Documents/prediction/datasets/NGSIM/us101/i101_trajectories-0750am-0805am.txt'
# us101_2 = '/home/sujith/Documents/prediction/datasets/NGSIM/us101/i101_trajectories-0805am-0820am.txt'
# us101_3 = '/home/sujith/Documents/prediction/datasets/NGSIM/us101/i101_trajectories-0820am-0835am.txt'
# i80_1 = '/home/sujith/Documents/prediction/datasets/NGSIM/i80/i80_trajectories-0400-0415.txt'
# i80_2 = '/home/sujith/Documents/prediction/datasets/NGSIM/i80/i80_trajectories-0500-0515.txt'
# i80_3 = '/home/sujith/Documents/prediction/datasets/NGSIM/i80/i80_trajectories-0515-0530.txt'

files = [us101_1, us101_2, us101_3, i80_1, i80_2, i80_3]

cols = ['dsID', 'vehID', 'frameNum', 'local_x', 'local_y', 'lane_id']
trajectories = []

print('Reading the data from raw text files')
for i, file in tqdm(enumerate(files)):
    df = pd.read_csv(file, header=None, delimiter=r'\s+')
    data_id = pd.DataFrame((i+1)*np.ones(df.shape[0], dtype=np.int32))
    df = pd.concat([data_id, df.iloc[:, [0, 1, 4, 5, 13]]], axis=1)
    # print(df.shape, df.dtypes)
    df.columns = cols
    if i <= 2:
        df.iloc[:, 5] = df.apply(lambda x: 6 if x[5]>=6 else x[5], axis=1)
    trajectories.append(df)

def parse_fields(traj):
    traj.loc[:, 'lat_man'] = np.ones(traj.shape[0], dtype=np.int32)
    traj.loc[:, 'long_man'] = np.ones(traj.shape[0], dtype=np.int32)
    grid = pd.DataFrame(np.zeros((traj.shape[0], 39), dtype=np.int32), columns=[str(i) for i in range(1, 40)])
    for k in range(traj.shape[0]):
        time = traj.iloc[k, 2]
        dsID = traj.iloc[k, 0]
        vehID = traj.iloc[k, 1]
        vehTraj = traj.loc[(traj.iloc[:, 0] == dsID) & (traj.iloc[:, 1] == vehID), :]
        ind = vehTraj[vehTraj.iloc[:, 2] == time].iloc[0,:].name
        lane = traj.iloc[k, 5]

        #Find lateral maneuver
        ub = min(vehTraj.shape[0]-1, ind+40)
        lb = max(0, ind-40)
        if vehTraj.iloc[ub, 5] > vehTraj.iloc[ind, 5] or vehTraj.iloc[ind, 5] > vehTraj.iloc[lb,5]:
            traj.loc[k, 'lat_man'] = 3
        elif vehTraj.iloc[ub, 5] < vehTraj.iloc[ind, 5] or vehTraj.iloc[ind, 5] < vehTraj.iloc[lb,5]:
            traj.loc[k, 'lat_man'] = 2
        else:
            traj.loc[k, 'lat_man'] = 1

        #Find longitudinal maneuver
        ub = min(vehTraj.shape[0]-1, ind+50)
        lb = max(0, ind-30)
        if ub == ind or lb == ind:
            traj.loc[k, 'long_man'] = 1
        else:
            v_hist = (vehTraj.iloc[ind, 4] - vehTraj.iloc[lb, 4])/(ind-lb)
            v_fut = (vehTraj.iloc[ub, 4] - vehTraj.iloc[ind, 4])/(ub-ind)
            if v_fut/(v_hist+0.00001) < 0.8:
                traj.loc[k, 'long_man'] = 2
            else:
                traj.loc[k, 'long_man'] = 1

        frameEgo = traj.loc[(traj.iloc[:, 0] == dsID) & (traj.iloc[:, 2] == time) & (traj.iloc[:, 5] == lane), :]
        # print(frameEgo)
        frameL = traj.loc[(traj.iloc[:, 0] == dsID) & (traj.iloc[:, 2] == time) & (traj.iloc[:, 5] == lane-1), :]
        frameR = traj.loc[(traj.iloc[:, 0] == dsID) & (traj.iloc[:, 2] == time) & (traj.iloc[:, 5] == lane+1), :]
        
        if not frameL.empty:
            for l in range(frameL.shape[0]):
                y = frameL.iloc[l, 4] - traj.iloc[k, 4]
                if abs(y) < 90:
                    gridInd = 1+round((y+90)/15)
                    grid.loc[k, str(gridInd)] = frameL.iloc[l, 1]
        for l in range(frameEgo.shape[0]):
            y = frameEgo.iloc[l, 4] - traj.iloc[k, 4]
            if abs(y) < 90 and y != 0:
                gridInd = 14+round((y+90)/15)
                grid.loc[k, str(gridInd)] = frameEgo.iloc[l, 1]
        if not frameR.empty:
            for l in range(frameR.shape[0]):
                y = frameR.iloc[l, 4] - traj.iloc[k, 4]
                if abs(y) < 90:
                    gridInd = 27+round((y+90)/15)
                    grid.loc[k, str(gridInd)] = frameR.iloc[l, 1]
    return pd.concat([traj, grid], axis=1)

print('Processing the dataframes in parallel with 6 workers')
p = mp.Pool(6)
trajectories = p.map(parse_fields, trajectories)
p.close()

trajAll = pd.concat([trajectories[i] for i in range(len(trajectories))], axis=0, ignore_index=True)

print('shape of trajAll: ', trajAll.shape)

trajTr = pd.DataFrame(columns=trajAll.columns)
trajVal = pd.DataFrame(columns=trajAll.columns)
trajTest = pd.DataFrame(columns=trajAll.columns)

print('Splitting into train, val and test datasets')
for i in range(1, 7):
    ul1 = round(0.7*trajAll.loc[trajAll.iloc[:, 0] == i, 1].max())
    ul2 = round(0.8*trajAll.loc[trajAll.iloc[:, 0] == i, 1].max())

    trajTr = pd.concat([trajTr, trajAll.loc[(trajAll.iloc[:, 0] == i) & (trajAll.iloc[:, 1] <= ul1), :]], axis=0, ignore_index=True)
    trajVal = pd.concat([trajVal, trajAll.loc[(trajAll.iloc[:, 0] == i) & (trajAll.iloc[:, 1] > ul1) & (trajAll.iloc[:, 1] <= ul2), :]], axis=0, ignore_index=True)
    trajTest = pd.concat([trajTr, trajAll.loc[(trajAll.iloc[:, 0] == i) & (trajAll.iloc[:, 1] > ul2), :]], axis=0, ignore_index=True)

print('Creating tracks')
for i in range(1, 7):
    trajSet = trajTr.loc[trajTr.iloc[:, 0] == i, :]
    carIds = list(trajSet.iloc[:, 1].unique())
    tracksTr = {(i, carIds[j]):trajSet.loc[trajSet.iloc[:, 1] == carIds[j], [2, 3, 4]].to_numpy() for j in carIds}

    trajSet = trajVal.loc[trajVal.iloc[:, 0] == i, :]
    carIds = list(trajSet.iloc[:, 1].unique())
    tracksVal = {(i, carIds[j]):trajSet.loc[trajSet.iloc[:, 1] == carIds[j], [2, 3, 4]].to_numpy() for j in carIds}

    trajSet = trajTest.loc[trajTest.iloc[:, 0] == i, :]
    carIds = list(trajSet.iloc[:, 1].unique())
    tracksTest = {(i, carIds[j]):trajSet.loc[trajSet.iloc[:, 1] == carIds[j], [2, 3, 4]].to_numpy() for j in carIds}

print('Keeping only tracks with more than 30 frames')
trajTr['keep'] = np.zeros(trajTr.shape[0], dtype=np.bool)
trajTr.loc[:, 'keep'] = trajTr.apply(lambda x: 1 if (tracksTr[(x[0], x[1])][0, 30] <= x[2]) & (tracksTr[(x[0], x[1])][0, -1] > x[2]) else 0, axis=1)
trajTr.loc[trajTr.loc[:, 'keep'] == 1, :].drop('keep', axis=1, inplace=True)
trajVal['keep'] = np.zeros(trajVal.shape[0], dtype=np.bool)
trajVal.loc[:, 'keep'] = trajVal.apply(lambda x: 1 if (tracksVal[(x[0], x[1])][0, 30] <= x[2]) & (tracksVal[(x[0], x[1])][0, -1] > x[2]) else 0, axis=1)
trajVal.loc[trajVal.loc[:, 'keep'] == 1, :].drop('keep', axis=1, inplace=True)
trajTest['keep'] = np.zeros(trajTest.shape[0], dtype=np.bool)
trajTest.loc[:, 'keep'] = trajTest.apply(lambda x: 1 if (tracksTest[(x[0], x[1])][0, 30] <= x[2]) & (tracksTest[(x[0], x[1])][0, -1] > x[2]) else 0, axis=1)
trajTest.loc[trajTest.loc[:, 'keep'] == 1, :].drop('keep', axis=1, inplace=True)

trainTraj = {}
trainTraj['traj'] = trajTr.to_numpy()
valTraj = {}
trainTraj['traj'] = trajVal.to_numpy()
testTraj = {}
testTraj['traj'] = trajTest.to_numpy()

print('saving into mat files')
savemat('trainTraj', trainTraj)
savemat('testTraj', testTraj)
savemat('valTraj', valTraj)

savemat('trainTracks', tracksTr)
savemat('valTracks', tracksVal)
savemat('testTracks', tracksTest)

print('Done!')