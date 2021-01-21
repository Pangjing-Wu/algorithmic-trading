import os
import re

import numpy as np


def load_m2t_train_log(file:str)->dict:
    episode = list()
    reward = list()
    vwap = list()
    market_vwap = list()
    f = open(file, mode='r')
    logs = f.readlines()
    f.close()
    for log in logs:
        episode += re.findall(r"Episode (.+?)/", log)
        reward += re.findall(r"train reward = (.+?),", log)
        vwap += re.findall(r"'vwap': (.+?),", log)
        market_vwap += re.findall(r"'market_vwap': (.+?)\}", log)
    for i in range(len(episode)):
        episode[i] = float(episode[i])
        reward[i] = float(reward[i])
        vwap[i] = float(vwap[i])
        market_vwap[i] = float(market_vwap[i])
    slippage = (np.array(vwap) - np.array(market_vwap)).tolist()
    reward25, reward50, reward75       = np.percentile(reward, [25, 50, 75])
    slippage25, slippage50, slippage75 = np.percentile(slippage, [25, 50, 75])
    extreme_index = list()
    for i, r, s in enumerate(zip(reward, slippage)):
        if r < reward50 and reward50 - r > 3 * (reward50 - reward25):
            extreme_index.append(i)
        elif r > reward50 and reward50 - r < 3 * (reward50 - reward75):
            extreme_index.append(i)
        elif s < slippage50 and slippage50 - s > 3 * (slippage50 - slippage25):
            extreme_index.append(i)
        elif s > slippage50 and slippage50 - s < 3 * (slippage50 - slippage75):
            extreme_index.append(i)
        else:
            pass
    ret = dict(episode=episode, reward=reward, vwap=vwap,
               market_vwap=market_vwap, slippage=slippage)
    for i in sorted(extreme_index,reverse=True):
        for k in ret.keys():
            ret[k].pop(i)
    return ret


def load_m3t_train_log(file:str)->dict:
    episode = list()
    in_reward = list()
    ex_reward = list()
    vwap = list()
    market_vwap = list()
    f = open(file, mode='r')
    logs = f.readlines()
    f.close()
    for log in logs:
        episode += re.findall(r"Episode (.+?)/", log)
        in_reward += re.findall(r"ave. intrinsic reward = (.+?),", log)
        ex_reward += re.findall(r"ave. extrinsic reward = (.+?),", log)
        vwap += re.findall(r"'vwap': (.+?),", log)
        market_vwap += re.findall(r"'market_vwap': (.+?)\}", log)
    for i in range(len(episode)):
        episode[i] = float(episode[i])
        in_reward[i] = float(in_reward[i])
        ex_reward[i] = float(ex_reward[i])
        vwap[i] = float(vwap[i])
        market_vwap[i] = float(market_vwap[i])
    slippage = (np.array(vwap) - np.array(market_vwap)).tolist()
    in_reward25, in_reward50, in_reward75 = np.percentile(in_reward, [25, 50, 75])
    ex_reward25, ex_reward50, ex_reward75 = np.percentile(ex_reward, [25, 50, 75])
    slippage25, slippage50, slippage75    = np.percentile(slippage, [25, 50, 75])
    extreme_index = list()
    for i, (in_r, ex_r, s) in enumerate(zip(in_reward, ex_reward, slippage)):
        if in_r < in_reward25 and in_reward50 - in_r > 3 * (in_reward50 - in_reward25):
            extreme_index.append(i)
        elif in_r > in_reward50 and in_reward50 - in_r < 3 * (in_reward50 - in_reward75):
            extreme_index.append(i)
        elif ex_r < ex_reward25 and ex_reward50 - ex_r > 3 * (ex_reward50 - ex_reward25):
            extreme_index.append(i)
        elif ex_r > ex_reward50 and ex_reward50 - ex_r < 3 * (ex_reward50 - ex_reward75):
            extreme_index.append(i)
        elif s < slippage50 and slippage50 - s > 3 * (slippage50 - slippage25):
            extreme_index.append(i)
        elif s > slippage50 and slippage50 - s < 3 * (slippage50 - slippage75):
            extreme_index.append(i)
        else:
            pass
    ret = dict(episode=episode, in_reward=in_reward, ex_reward=ex_reward,
               vwap=vwap, market_vwap=market_vwap, slippage=slippage)
    for i in sorted(extreme_index,reverse=True):
        for k in ret.keys():
            ret[k].pop(i)
    return ret


def read_m2t_train_args(filename:str):
    filename = filename.rstrip('.log')
    filename = filename.split('-')
    ret = dict(stock=filename[0], model=filename[1],
               eps=float(filename[2].strip('eps')) / 10,
               reward_type=filename[3],
               quote_len=filename[4].strip('len'),
               i=filename[5], n=filename[6])
    return ret


def read_m3t_train_args(filename:str):
    filename = filename.rstrip('.log')
    filename = filename.split('-')
    ret = dict(stock=filename[0],
               model= filename[1],
               eps=float(filename[2].strip('eps')) / 10,
               reward_type=filename[3],
               quote_len=filename[4].strip('len'))
    return ret