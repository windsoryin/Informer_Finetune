import numpy as np
# 6/20add code
from fastdtw import fastdtw


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# 6/20add code
def DTW(pred, true):
    distance, path = fastdtw(pred, true)
    return distance

# 6/20add code
def Temporal(pred, true, N_output):
    tdi = 0
    distance, path = fastdtw(pred, true)
    Dist = 0
    for i, j in path:
        Dist += (i - j) * (i - j)
    tdi += Dist / (N_output * N_output)
    return tdi

def Score(pred_all,true_all,th1,th2):
    score=[0,0,0,0]
    ####230809: TS score
    # predict threshhodl : th1
    # true threshhodl : th2
    for itr in range(pred_all.shape[0]):
        pred=pred_all[itr,:,:]
        true=true_all[itr,:,:]
        index1 = np.sum(pred.numpy() > th1)
        index2 = np.sum(true.numpy() > th2)
        if index1 > 0 and index2 > 0:
            score[0] = score[0] + 1 #tp
        if index1 > 0 and index2 == 0:
            score[1] = score[1] + 1 #fp
        if index1 == 0 and index2 > 0:
            score[2] = score[2] + 1 #fn
        if index1 == 0 and index2 == 0:
            score[3] = score[3] + 1 #tn
    return score
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe