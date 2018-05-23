#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 5/19/18 7:29 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''
from __future__ import print_function
from __future__ import division
import ConfigParser
from environment import GridRoom
from time import *
import numpy as np
from copy import *
from agent import *
import cPickle

config = ConfigParser.ConfigParser()
with open('config.cfg', 'rw') as cfgfile:
    config.readfp(cfgfile)
    _width = int(config.get('environ', 'Width'))
    _height = int(config.get('environ', 'Height'))
    _num_players = int(config.get('environ', 'Players'))
    _ammo = int(config.get('environ', 'Ammo'))
    _max_step = int(config.get('environ', 'Max_Step'))
    _gamma = float(config.get('environ', 'Gamma'))


def train():
    world = GridRoom()
    players = [RMAgent(i) for i in range(0, _num_players)]
    begin = time()
    # sampled_exp = []
    for _iteration in range(10000):
        sampled_exp = []
        iter_time = time()
        # for _pid in range(1, _num_players):
        #     players[_pid].u_s = copy(players[0].u_s)
        #     players[_pid].u_sa = copy(players[0].u_sa)
        #     players[_pid].average_strategy = copy(players[0].average_strategy)
        for _i in range(500):
            for _pid in range(_num_players):
                players[_pid].set_ammo(_num_players - 1)
                players[_pid].exp_buffer = []
            ob = world.cur_state()
            done = False
            prev_j_ac = [9] * _num_players
            while not done:
                joint_action = [9] * _num_players
                for _pid in world.alive_players:
                    _ac = players[_pid].action([ob, prev_j_ac], players[_pid].valid_action())
                    if _ac == 0:
                        players[_pid].set_ammo(players[_pid].ammo - 1)
                    joint_action[_pid] = _ac
                done, r, n_state = world.step(joint_action)
                for _pid in world.alive_players:
                    players[_pid].exp_buffer.append([str(_pid) + ''.join(map(str, ob)) + str(players[_pid].ammo) + ''.join(map(str, prev_j_ac)), joint_action[_pid], r[_pid]])
                prev_j_ac = copy(joint_action)
                ob = n_state
            for _pid in range(1):  # TODO : replace with player 1
                v = 0
                for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                    players[_pid].exp_buffer[_k][2] += _gamma * v
                    v = players[_pid].exp_buffer[_k][2]
                sampled_exp.extend(players[_pid].exp_buffer)  # all agents share it
            world.reset()
        print('Episode time: %.2f' % (time() - iter_time))
        # end sample
        players[0].update_policy(sampled_exp)
    print('Time eplapsed: %.2f' % (time() - begin))
    # print(sampled_exp)
    with open('v.pkl', 'wb') as f:
        cPickle.dump(players[0].u_s, f, 2)
    with open('q.pkl', 'wb') as f:
        cPickle.dump(players[0].u_sa, f, 2)
    with open('pi.pkl', 'wb') as f:
        cPickle.dump(players[0].average_strategy, f, 2)


def test():
    world = GridRoom()
    players = [RMAgent(0)]
    for i in range(1, _num_players):
        players.append(RandomAgent(i))
    players[0].test = True
    begin = time()
    # with open('v.pkl', 'rb') as f:
    #     players[0].u_s = cPickle.load(f)
    # with open('q.pkl', 'rb') as f:
    #     players[0].u_sa = cPickle.load(f)
    # with open('pi.pkl', 'rb') as f:
    #     players[0].average_strategy = cPickle.load(f)
    total_r = np.zeros(_num_players)
    print('Time for load model: ', time()-begin, players[0].u_s.__len__(), players[0].u_sa.__len__(), players[0].average_strategy.__len__())
    begin = time()
    # sampled_exp = []
    for _iteration in range(100000):
        iter_time = time()
        for _i in range(1):
            for _pid in range(_num_players):
                players[_pid].set_ammo(_num_players - 1)
            ob = world.cur_state()
            done = False
            prev_j_ac = [9] * _num_players
            while not done:
                joint_action = [9] * _num_players
                for _pid in world.alive_players:
                    _ac = players[_pid].action([ob, prev_j_ac], players[_pid].valid_action())
                    if _ac == 0:
                        players[_pid].set_ammo(players[_pid].ammo - 1)
                    joint_action[_pid] = _ac
                done, r, n_state = world.step(joint_action)
                prev_j_ac = copy(joint_action)
                ob = n_state
            total_r = np.add(total_r, world.players_total_reward)
            world.reset()
        # print('Episode time: %.2f' % (time() - begin))
    print('Time eplapsed: %.2f min' % ((time() - begin)/60.0))
    print(total_r)
    print('unseen, seen', players[0].unseen, players[0].seen)
    # for _key, _item in players[0].u_s.iteritems():
    #     print(_key, _item)
    # print('------------')
    # for _key, _item in players[0].u_sa.iteritems():
    #     print(_key, _item)
    # print('------------')
    # for _k, _i in players[0].average_strategy.iteritems():
    #     if _i.count(0) < _i.__len__()-1:
    #         print(_k, _i)


if __name__ == '__main__':
    train()
    # test()

