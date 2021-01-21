import torch


average = lambda x: sum(x) / len(x) if len(x) else 0


class HRLTrader(object):

    def __init__(self, macro_model, micro_model):
        self.__macro_model = macro_model
        self.__micro_model = micro_model

    def __call__(self, env):
        step       = 0
        subgoals   = list()
        ex_rewards = list()
        in_rewards = list()
        ex_s = env.reset()
        while not env.final:
            goal = torch.argmax(self.__macro_model(ex_s)).item()
            subgoals.append(goal)
            in_s = env.update_subgoal(goal)
            # print('goal: %s' % env._subgoal)
            while not env.subfinal:
                a = torch.argmax(self.__micro_model(in_s, goal)).item()
                in_s1, in_r = env.step(a)
                in_rewards.append(in_r)
                step += 1
                # print('current filled %s' % env._filled)
            # print('subfilled: %s' % env._subfilled)
            ex_s, ex_r = env.extrinsic_state, env.extrinsic_reward
            ex_rewards.append(ex_r)
        ex_reward, in_reward = average(ex_rewards), average(in_rewards)
        ret = dict(ex_reward=ex_reward, in_reward=in_reward,
                   subgoals=subgoals, step=step, metrics=env.metrics())
        return ret