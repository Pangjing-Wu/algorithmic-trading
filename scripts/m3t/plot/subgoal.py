import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = [600000, 600010, 600018, 600028, 600030, 600048, 600050, 600104]
colors = [(13/255,72/255,148/255), [], [], (34/255,107/255,178/255), [], [],
          (79/255,151/255,200/255), (129/255,179/255,216/255), (181/255,211/255,233/255)]
ylims  = [(0.12, 0.14), (0.125,0.135), (0.13,0.1601), (0.10,0.135), (0.10,0.1151), (0.128,0.134), (0.11,0.14), (0.11,0.14)]
fig, ax = plt.subplots(2, 4, figsize=(20, 6))
for ax1, stock, ylim in zip(ax.flatten(),stocks, ylims):
    df = pd.read_csv('./scripts/m3t/plot/subgoals/%d.csv' % stock, index_col='tranche').sort_index(axis=1)
    subgoal = df.drop('speed', axis=1)
    bottom = None
    for col in subgoal.columns:
        if bottom is None:
            ax1.bar(subgoal.index, subgoal[col], color=colors[int(col[-1])], align="center", label=col)
            bottom = subgoal[col]
        else:
            ax1.bar(subgoal.index, subgoal[col], color=colors[int(col[-1])],
                    bottom=bottom, align="center", label=col)
            bottom += subgoal[col]
    ax1.set_ylim(0, 500)
    ax1.set_xlabel(stock)
    ax1.set_ylabel('Subgoals')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['speed'], linestyle='--', color='gray', marker='o', label='speed')
    ax2.set_ylim(*ylim)
    ax2.set_yticks(np.arange(ylim[0], ylim[1], 0.005))
    ax2.set_ylabel('Speed')
    ax1.legend(loc=2)
    ax2.legend(loc=1)
plt.tight_layout()
plt.savefig('./scripts/m3t/plot/subgoals.png')