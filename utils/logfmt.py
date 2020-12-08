import os
import re

import numpy as np


def load_m3t_train_log(file:str)->dict:
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
    slippage = np.array(vwap) - np.array(market_vwap)
    ret = dict(episode=episode, reward=reward, vwap=vwap,
               market_vwap=market_vwap, slippage=slippage)
    return ret


def load_hrl_train_log(file:str)->dict:
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
    slippage = np.array(vwap) - np.array(market_vwap)
    ret = dict(episode=episode, in_reward=in_reward, ex_reward=ex_reward,
               vwap=vwap, market_vwap=market_vwap, slippage=slippage)
    return ret


def read_m3t_train_args(filename:str):
    filename = filename.rstrip('.log')
    filename = filename.split('-')
    ret = dict(stock=filename[0], model=filename[1],
               eps=float(filename[2].strip('eps')) / 10,
               reward_type=filename[3],
               quote_len=filename[4].strip('len'),
               i=filename[5], n=filename[6])
    return ret


def read_hrl_train_args(filename:str):
    filename = filename.rstrip('.log')
    filename = filename.split('-')
    ret = dict(stock=filename[0], 
               eps=float(filename[1].strip('eps')) / 10,
               reward_type=filename[2],
               quote_len=filename[3].strip('len'))
    return ret