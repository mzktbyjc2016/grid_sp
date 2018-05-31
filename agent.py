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
from tfmodel import *
import ConfigParser

config = ConfigParser.ConfigParser()
with open('config.cfg', 'rw') as cfgfile:
    config.readfp(cfgfile)
    _width = int(config.get('environ', 'Width'))
    _height = int(config.get('environ', 'Height'))
    _num_players = int(config.get('environ', 'Players'))
    _ammo = int(config.get('environ', 'Ammo'))
    _max_step = int(config.get('environ', 'Max_Step'))
    _gamma = float(config.get('environ', 'Gamma'))
    _train_iter = int(config.get('algorithm', 'train_iter'))
    _sample_iter = int(config.get('algorithm', 'sample_iter'))
    _test_iter = int(config.get('algorithm', 'test_iter'))
    # _s_th = int(config.get('algorithm', 'simulation_thread'))
    _frame_stack = int(config.get('environ', 'Frame_stack'))


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
        self._iter = 0
        self.exploration = False
        self.wtk = set()

    def parse_state(self, state):
        pass

    def action(self, state, action_space):
        # return choice(action_space)
        if not self.test:
            state_key = str(self.id)+''.join(map(str, state[0]))+str(self.ammo)+''.join(map(str, state[1]))
            if self.exploration:
                return choice(action_space)
            if state_key not in self.u_s:
                self.unseen += 1
                self.wtk.add(state_key)
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
                    # print(prob)
                    # if prob.tolist().count(0) < len(action_space) - 1:
                    #     print(self.id, prob)
                    # if np.random.random() < 1e-1:
                    #     print(prob)
                    # print(prob)
                else:
                    prob = np.true_divide(np.ones(len(action_space)), len(action_space))  # uniformly if no regret
                return np.random.choice(action_space, p=prob)
        else:
            state_key = str(self.id) + ''.join(map(str, state[0])) + str(self.ammo) + ''.join(map(str, state[1]))
            if state_key in self.average_strategy:
                frequency = self.average_strategy[state_key]
                if len(action_space) > 4:
                    prob = np.true_divide(frequency, sum(frequency))
                    if state_key not in self.wtk and prob[0] > 0.2:
                        with open('{}.prob'.format(self.id), 'ab+') as f:
                            f.write(state_key + ' ' + str(prob)+'\n')
                        self.wtk.add(state_key)
                else:
                    prob = np.true_divide(frequency[1:], sum(frequency[1:]))
                self.seen += 1
                # if np.random.random() < 1e-3:
                # if prob.tolist().count(0) < len(action_space) - 1 or prob[0] == 1:
                #     print(self.id, prob)
                return np.random.choice(action_space, p=prob)
            else:
                self.wtk.add(state_key)
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
        print('iter: ', self._iter)
        for _item in experience:  # each item is [[id, ob, ammo, prev_joint_action], action, return]
            _s_key = str(_item[0][0]) + ''.join(map(str, _item[0][1])) + str(_item[0][2]) + ''.join(map(str, _item[0][3]))
            if _s_key not in self.u_s:
                self.u_s[_s_key] = [_item[2], 1]
            else:
                self.u_s[_s_key][0] += _item[2]
                self.u_s[_s_key][1] += 1
            s_a_key = _s_key + str(_item[1])
            if s_a_key not in self.u_sa:
                self.u_sa[s_a_key] = [_item[2], 1]
            else:
                self.u_sa[s_a_key][0] += _item[2]
                self.u_sa[s_a_key][1] += 1
            if _s_key not in self.average_strategy:
                ac_v = [0]*5
                ac_v[_item[1]] = 1
                self.average_strategy[_s_key] = ac_v
            else:
                self.average_strategy[_s_key][_item[1]] += 1


class RandomAgent(RMAgent):

    def action(self, state, action_space):
        return choice(action_space)


class ShootingAgent(RMAgent):

    def action(self, state, action_space):
        # state_key = str(self.id) + ''.join(map(str, state[0])) + str(self.ammo) + ''.join(map(str, state[1]))
        if state[1][0] == 0:
            if state[0][0] == state[0][3] and state[0][1] == state[0][4] or state[0][0] != state[0][3] and state[0][1] != state[0][4]:
                if state[0][3] == 0 and state[0][4] == 0:
                    return 3
                elif state[0][3] == 0 and state[0][4] == 1:
                    return 2
                elif state[0][3] == 1 and state[0][4] == 0:
                    return 1
                else:
                    return 4
            else:
                if state[0][3] == 0:
                    return 4
                if state[0][3] == 1:
                    return 3
        elif self.ammo > 0 and np.random.random() < 1:
            return 0
        else:
            return choice(range(1, 5))


class ShootingAgent1(RMAgent):

    def action(self, state, action_space):
        # state_key = str(self.id) + ''.join(map(str, state[0])) + str(self.ammo) + ''.join(map(str, state[1]))
        if state[1][0] == 0:
            if state[0][0] == state[0][3] and state[0][1] == state[0][4] or state[0][0] != state[0][3] and state[0][1] != state[0][4]:
                if state[0][3] == 0 and state[0][4] == 0:
                    return 3
                elif state[0][3] == 0 and state[0][4] == 1:
                    return 2
                elif state[0][3] == 1 and state[0][4] == 0:
                    return 1
                else:
                    return 4
            else:
                if state[0][3] == 0:
                    return 4
                if state[0][3] == 1:
                    return 3
        elif self.ammo > 0:
            if not self.is_in_front([state[0][3], state[0][4]], state[0][5], [state[0][0], state[0][1]]) and np.random.random() < 1:
                return 0
            else:
                return choice(range(1, 5))
        else:
            return choice(range(1, 5))

    @staticmethod
    def is_in_front(pos1, dir1, pos2):
        """
        use this function to judge whether the position2 is in front of the ammo
        :param pos1: some ammo's position
        :param dir1: some ammo's direction
        :param pos2: some player's position
        :return: Boolean
        """
        if (pos1[0]-pos2[0])*(pos1[1]-pos2[1]) != 0:  # not in the same horizontal or vertical line
            return False
        else:
            if pos1[0] == pos2[0]:  # in the same row and will return False when pos1 coincides with pos2
                if dir1 == 3 or dir1 == 4:
                    return False
                return np.dot((0, pos2[1] - pos1[1]), (0, (-1)**dir1)) > 0
            else:  # in the same column
                if dir1 == 1 or dir1 ==2:
                    return False
                return np.dot((pos2[0]-pos1[0], 0), ((-1)**dir1, 0)) > 0


def one_hot(_index, dim):
    _tmp = [0]*dim
    _tmp[_index] = 1
    return _tmp


class NRMAgent(RMAgent):
    """
    This is a neural agent that uses mlp only and the input state is encoded as follow
        ID(1)+ (one hot pos (N+M) + one hots dir (4))* # of players+ ammo(1, [0,1] float)*# of repetition + one hot prev_joint_action (# of players*5)
        (10+N+M)* # of players + repetition
    """
    def __init__(self, player_id=0):
        super(NRMAgent, self).__init__(player_id)
        # TODO overload policy and value function
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        model = Model(self.graph)
        self.out, self.t_out, self.average_strategy, self.behavior_pi, self.update_weights = model.create_model(self.session, input_dim=(10 + _height + _width) * _num_players + 3, out_dim=11, action_dim=5)
        # self.out2, self.update_weights = model.create_target_model(input_dim=(10 + _height + _width) * _num_players + 3, out_dim=11, action_dim=5)
        # self.session.run(tf.local_variables_initializer())
        # self.session.run(tf.global_variables_initializer())
        # tf.reset_default_graph()
        with self.graph.as_default():
            weights = self.session.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pi'))
        for w in weights:
            print(np.shape(w))
        print('???', tf.trainable_variables())
        self.update_weights(weights)

    def action(self, state, action_space):
        one_hot_state = self.parse_state(state)
        s = one_hot(self.id, _num_players) + one_hot_state + [self.ammo/float(_ammo)]*3
        prob = self.behavior_pi(s)
        if len(action_space) > 4:
            return np.random.choice(action_space, p=prob)
        else:
            trunc_prob = np.true_divide(prob[1:], sum(prob[1:]))
            return np.random.choice(action_space, p=trunc_prob)

    def update_policy(self, experience):
        
        for _item in experience:  # each item is [[id, ob, ammo, prev_joint_action], action, return]
            one_hot_state = []
            one_hot(_item[0], _num_players)
        np.array(experience)

    def parse_state(self, state):
        one_hot_state = []
        # print(state[0], state[1])
        for i in range(len(state[0])):
            if i % 3 == 0:
                one_hot_state += one_hot(state[0][i], _width)
            elif i % 3 == 1:
                one_hot_state += one_hot(state[0][i], _height)
            else:
                one_hot_state += one_hot(state[0][i]-1, 4)
        for i in range(len(state[1])):
            if state[1][i] != 9:
                one_hot_state += one_hot(state[1][i], 5)
            else:
                one_hot_state += [0]*5
        return one_hot_state
