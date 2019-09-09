import pandas as pd
import numpy as np
import multiprocessing as mp

us101_1 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0750am-0805am/trajectories-0750am-0805am.txt'
us101_2 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0805am-0820am/trajectories-0805am-0820am.txt'
us101_3 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/us101/0820am-0835am/trajectories-0820am-0835am.txt'
i80_1 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0400pm-0415pm/trajectories-0400-0415.txt'
i80_2 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0500pm-0515pm/trajectories-0500-0515.txt'
i80_3 = '/home/sujith/Documents/Projects/ADS/prediction/datasets/NGSIM/i80/0515pm-0530pm/trajectories-0515-0530.txt'

files = [us101_1, us101_2, us101_3, i80_1, i80_2, i80_3]
cols = ['dsID', 'vehID', 'frameNum', 'local_x', 'local_y', 'lane_id']
trajectories = []

for i, file in enumerate(files):
    df = pd.read_csv(file, header=None, delimiter='\s+')
    data_id = pd.DataFrame((i+1)*np.ones(df.shape[0], dtype=np.int32))
    df = pd.concat([data_id, df.iloc[:, [0, 1, 4, 5, 13]]], axis=1)
    if i <= 2:
        df.ix[:, 5] = df.ix[:, 5].apply(lambda x: 6 if x>=6 else x)
    df.columns = cols
    trajectories.append(df)

def parse_fields(traj):
    traj.loc[:, 'lat_man'] = np.ones(traj.shape[0], dtype=np.int32)
    traj.loc[:, 'long_man'] = np.ones(traj.shape[0], dtype=np.int32)
    grid = pd.DataFrame(np.zeros((traj.shape[0], 39), dtype=np.int32), columns=[str(i) for i in range(1, 40)])
    for k in range(traj.shape[0]):
        time = traj.loc[k, 'frameNum']
        dsID = traj.loc[k, 'dsID']
        vehID = traj.loc[k, 'vehID']
        vehTraj = traj.loc[(traj.loc[:, 'dsID'] == dsID) & (traj.loc[:, 'vehID'] == vehID), :]
        ind = vehTraj[vehTraj['frameNum'] == time].iloc[0,:].name
        lane = traj.loc[k, 'lane_id']

        #Find lateral maneuver
        ub = min(vehTraj.shape[0], ind+40)
        lb = max(0, ind-40)
        if vehTraj.iloc[ub, 'lane_id'] > vehTraj.iloc[ind, 'lane_id'] | vehTraj.iloc[ind, 'lane_id'] > vehTraj.iloc[lb,'lane_id']:
            traj.loc[k, 'lat_man'] = 3
        elif vehTraj.iloc[ub, 'lane_id'] < vehTraj.iloc[ind, 'lane_id'] | vehTraj.iloc[ind, 'lane_id'] < vehTraj.iloc[lb,'lane_id']:
            traj.loc[k, 'lat_man'] = 2
        else:
            traj.loc[k, 'lat_man'] = 1

        #Find longitudinal maneuver
        ub = min(vehTraj.shape[0], ind+50)
        lb = max(1, ind-30)
        if ub == ind | lb == ind:
            traj.loc[k, 'long_man'] = 1
        else:
            v_hist = (vehTraj.loc[ind, 'local_y'] - vehTraj.loc[lb, 'local_y'])/(ind-lb)
            v_fut = (vehTraj.loc[ub, 'local_y'] - vehTraj.loc[ind, 'local_y'])/(ub-ind)
            if v_fut/(v_hist+0.00001) < 0.8:
                traj.loc[k, 'long_man'] = 2
            else:
                traj.loc[k, 'long_man'] = 1

        frameEgo = traj.loc[(traj.loc[:, 'dsID'] == dsID) & (traj.loc[:, 'frameNum'] == time) & (traj.loc[:, 'lane_id'] == lane), :]
        frameL = traj.loc[(traj.loc[:, 'dsID'] == dsID) & (traj.loc[:, 'frameNum'] == time) & (traj.loc[:, 'lane_id'] == lane-1), :]
        frameR = traj.loc[(traj.loc[:, 'dsID'] == dsID) & (traj.loc[:, 'frameNum'] == time) & (traj.loc[:, 'lane_id'] == lane+1), :]
        
        if frameL:
            for l in range(frameL.shape[0]):
                y = frameL.loc[l, 'local_y'] - traj.loc[k, 'local_y']
                if abs(y) < 90:
                    gridInd = 1+round((y+90)/15)
                    grid.loc[k, str(gridInd)] = frameL.loc[l, 'vehID']
        for l in range(frameEgo.shape[0]):
            y = frameEgo.loc[l, 'local_y'] - traj.loc[k, 'local_y']
            if abs(y) < 90 & y != 0:
                gridInd = 14+round((y+90)/15)
                grid.loc[k, str(gridInd)] = frameEgo.loc[l, 'vehID']
        if frameR:
            for l in range(frameR.shape[0]):
                y = frameR.loc[l, 'local_y'] - traj.loc[k, 'local_y']
                if abs(y) < 90:
                    gridInd = 27+round((y+90)/15)
                    grid.loc[k, str(gridInd)] = frameR.loc[l, 'vehID']
    return pd.concat([traj, grid], axis=1)



