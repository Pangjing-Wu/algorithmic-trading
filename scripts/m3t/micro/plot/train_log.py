import os
import sys

import matplotlib.pyplot as plt

sys.path.append('./')
from utils.logfmt import load_m3t_train_log, read_m3t_train_args


def compare_parameters(filepath, savepath, n_tranche, figtype='eps'):
    os.makedirs(savepath, exist_ok=True)
    n_tranche = list(range(1, n_tranche+1))
    for i in n_tranche:
        files = [f for f in os.listdir(filepath) if f[-7] == str(i)]
        fig, ax = plt.subplots(len(files),2, figsize=(7,15))
        for j, file in enumerate(files):
            info = load_m3t_train_log(os.path.join(filepath, file))
            args = read_m3t_train_args(file)
            ax[j,0].plot(info['episode'], info['reward'])
            ax[j,0].set_title('%s' % file.rstrip('.log'), fontsize=7)
            ax[j,0].set_xlabel('episode', fontsize=7)
            ax[j,0].set_ylabel('accum. reward per episode', fontsize=7)
            ax[j,0].xaxis.set_major_locator(plt.MultipleLocator(2000))
            ax[j,0].xaxis.set_minor_locator(plt.MultipleLocator(200))
            ax[j,0].yaxis.set_major_locator(plt.MultipleLocator(500))
            ax[j,0].yaxis.set_minor_locator(plt.MultipleLocator(50))
            ax[j,0].tick_params(labelsize=7)
            ax[j,0].grid(which='major', axis='both', linewidth=0.75, linestyle='-', color='lightgray')
            ax[j,0].grid(which='minor', axis='both', linewidth=0.25, linestyle='-', color='lightgray')
            ax[j,1].plot(info['episode'], info['slippage'])
            ax[j,1].set_title('%s' % file.rstrip('.log'), fontsize=7)
            ax[j,1].set_xlabel('episode', fontsize=7)
            ax[j,1].set_ylabel('slippage', fontsize=7)
            ax[j,1].xaxis.set_major_locator(plt.MultipleLocator(2000))
            ax[j,1].xaxis.set_minor_locator(plt.MultipleLocator(200))
            ax[j,1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax[j,1].yaxis.set_minor_locator(plt.MultipleLocator(0.01))
            ax[j,1].tick_params(labelsize=7)
            ax[j,1].grid(which='major', axis='both', linewidth=0.75, linestyle='-', color='lightgray')
            ax[j,1].grid(which='minor', axis='both', linewidth=0.25, linestyle='-', color='lightgray')
        plt.tight_layout()
        savename = '%s-%s-%s.%s' %(args['stock'], args['i'], args['n'], figtype)
        plt.savefig(os.path.join(savepath, savename))


if __name__ == '__main__':
    filepath  = './results/logs/vwap/m3t/micro'
    savepath = './results/outputs/vwap/m3t/micro'
    compare_parameters(filepath, savepath, 4, figtype='eps')