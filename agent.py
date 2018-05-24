#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 5/19/18 7:29 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''

from random import *
import numpy as np


class Agent(object):

    def __init__(self):
        pass

    def action(self, state, action_space):
        pass

    def valid_action(self):
        pass


class RMAgent(Agent):

    def __init__(self, player_id=0):
        super(RMAgent, self).__init__()
        self.id = player_id
        self.u_s = {}
        self.u_sa = {}
        self.average_strategy = {}
        self.exp_buffer = []
        self.ammo = 9999
        self.test = False
        self.unseen = 0
        self.seen = 0

    def action(self, state, action_space):
        # return choice(action_space)
        if not self.test:
            state_key = str(self.id)+''.join(map(str, state[0]))+str(self.ammo)+''.join(map(str, state[1]))
            if state_key not in self.u_s:
                self.unseen += 1
                return choice(action_space)
            else:
                _imm_regret = []
                var_u_s = self.u_s[state_key]
                exp_u = var_u_s[0]/var_u_s[1]
                for _ac in action_space:
                    s_a_key = state_key+str(_ac)
                    if s_a_key not in self.u_sa:
                        exp_u_a = 0.0
                    else:
                        var_u_sa = self.u_sa[s_a_key]
                        exp_u_a = var_u_sa[0]/var_u_sa[1]
                    _imm_regret.append(exp_u_a-exp_u)
                regret_plus = np.maximum([0] * len(_imm_regret), _imm_regret)
                self.seen += 1
                if np.sum(regret_plus) > 0:
                    # return np.argmax(regret_plus)
                    prob = np.true_divide(regret_plus, np.sum(regret_plus))  # act according to regret
                    # if np.random.random() < 1e-1:
                    #     print(prob)
                else:
                    prob = np.true_divide(np.ones(len(action_space)), len(action_space))  # uniformly if no regret
                return np.random.choice(action_space, p=prob)
        else:
            state_key = str(self.id) + ''.join(map(str, state[0])) + str(self.ammo) + ''.join(map(str, state[1]))
            if state_key in self.average_strategy:
                frequency = self.average_strategy[state_key]
                if len(action_space) > 4:
                    prob = np.true_divide(frequency, sum(frequency))
                else:
                    prob = np.true_divide(frequency[1:], sum(frequency[1:]))
                self.seen += 1
                # if np.random.random() < 1e-3:
                #     print(prob)
                return np.random.choice(action_space, p=prob)
            else:
                self.unseen += 1
                return choice(action_space)

    def valid_action(self):
        if self.ammo == 0:
            return range(1, 5)
        else:
            return range(5)

    def set_ammo(self, amount):
        self.ammo = amount

    def update_policy(self, experience):
        for _item in experience:  # each item is [state, action, return]
            if _item[0] not in self.u_s:
                self.u_s[_item[0]] = [_item[2], 1]
            else:
                self.u_s[_item[0]][0] += _item[2]
                self.u_s[_item[0]][1] += 1
            s_a_key = _item[0] + str(_item[1])
            if s_a_key not in self.u_sa:
                self.u_sa[s_a_key] = [_item[2], 1]
            else:
                self.u_sa[s_a_key][0] += _item[2]
                self.u_sa[s_a_key][1] += 1
            if _item[0] not in self.average_strategy:
                ac_v = [0]*5
                ac_v[_item[1]] = 1
                self.average_strategy[_item[0]] = ac_v
            else:
                self.average_strategy[_item[0]][_item[1]] += 1


class RandomAgent(RMAgent):

    def action(self, state, action_space):
        return choice(action_space)
