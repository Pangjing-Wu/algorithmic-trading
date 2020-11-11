from typing import List, Union
import pandas as pd

'''
这边的模块这样改：
   输入日期和Marco trader的类型，从缓冲中读取volume profile
   若不存在该缓存，则调用训练模块
'''


argmax = lambda a: [i for i, val in enumerate(a) if (val == max(a))][0]


def distribute_task(goal:int, profile:pd.DataFrame):
    ratio = [v / (profile['volume'].sum() + 1e-8) for v in profile['volume']]
    subgoals = [int(goal * r // 100 * 100) for r in ratio]
    subgoals[argmax(subgoals)] += goal - sum(subgoals)
    tasks = pd.DataFrame(dict(start=profile['start'], end=profile['end'], goal=subgoals))
    return tasks