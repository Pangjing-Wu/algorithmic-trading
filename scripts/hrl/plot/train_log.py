import os
import sys

import matplotlib.pyplot as plt

sys.path.append('./')
from utils.logfmt import load_hrl_train_log


def compare_parameters(filepath, savepath, n_tranche, figtype='eps'):
    os.makedirs(savepath, exist_ok=True)
    files = sorted([f for f in os.listdir(filepath) if f[13:18] != 'Atten'])
    fig, ax = plt.subplots(len(files), 3, figsize=(7,20))
    for j, file in enumerate(files):
        info = load_hrl_train_log(os.path.join(filepath, file))
        ax[j,0].plot(info['episode'], info['ex_reward'])
        ax[j,0].set_title('%s' % file.rstrip('.log'), fontsize=7)
        ax[j,0].set_xlabel('episode', fontsize=7)
        ax[j,0].set_ylabel('ave. ex. r per episode', fontsize=7)
        ax[j,0].xaxis.set_major_locator(plt.MultipleLocator(2000))
        ax[j,0].xaxis.set_minor_locator(plt.MultipleLocator(200))
        ax[j,0].yaxis.set_major_locator(plt.MultipleLocator(200))
        ax[j,0].yaxis.set_minor_locator(plt.MultipleLocator(20))
        ax[j,0].tick_params(labelsize=7)
        ax[j,0].grid(which='major', axis='both', linewidth=0.75, linestyle='-', color='lightgray')
        ax[j,0].grid(which='minor', axis='both', linewidth=0.25, linestyle='-', color='lightgray')
        ax[j,1].plot(info['episode'], info['in_reward'])
        ax[j,1].set_title('%s' % file.rstrip('.log'), fontsize=7)
        ax[j,1].set_xlabel('episode', fontsize=7)
        ax[j,1].set_ylabel('ave. in. r  per episode', fontsize=7)
        ax[j,1].xaxis.set_major_locator(plt.MultipleLocator(2000))
        ax[j,1].xaxis.set_minor_locator(plt.MultipleLocator(200))
        ax[j,1].yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax[j,1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax[j,1].tick_params(labelsize=7)
        ax[j,1].grid(which='major', axis='both', linewidth=0.75, linestyle='-', color='lightgray')
        ax[j,1].grid(which='minor', axis='both', linewidth=0.25, linestyle='-', color='lightgray')
        ax[j,2].plot(info['episode'], info['slippage'])
        ax[j,2].set_title('%s' % file.rstrip('.log'), fontsize=7)
        ax[j,2].set_xlabel('episode', fontsize=7)
        ax[j,2].set_ylabel('slippage', fontsize=7)
        ax[j,2].xaxis.set_major_locator(plt.MultipleLocator(2000))
        ax[j,2].xaxis.set_minor_locator(plt.MultipleLocator(200))
        ax[j,2].yaxis.set_major_locator(plt.MultipleLocator(0.005))
        ax[j,2].yaxis.set_minor_locator(plt.MultipleLocator(0.0005))
        ax[j,2].tick_params(labelsize=7)
        ax[j,2].grid(which='major', axis='both', linewidth=0.75, linestyle='-', color='lightgray')
        ax[j,2].grid(which='minor', axis='both', linewidth=0.25, linestyle='-', color='lightgray')
    plt.tight_layout()
    savename = '%s.%s' %('train', figtype)
    plt.savefig(os.path.join(savepath, savename))


if __name__ == '__main__':
    filepath  = './results/logs/vwap/hrl'
    savepath = './results/outputs/vwap/hrl'
    compare_parameters(filepath, savepath, 4, figtype='eps')