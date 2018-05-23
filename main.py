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
from multiprocessing import Process, Lock, Queue, Pool

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
    _s_th = int(config.get('algorithm', 'simulation_thread'))


def simulation(players, q):
    world = GridRoom()
    sampled_exp = [[]]*_num_players
    for _i in range(_sample_iter):
        for _pid in range(_num_players):
            players[_pid].set_ammo(_ammo)
            players[_pid].exp_buffer = []
        ob = world.cur_state()
        done = False
        prev_j_ac = [9] * _num_players
        while not done:
            joint_action = [9] * _num_players
            for _pid in world.alive_players:
                _ac = players[_pid].action([ob, prev_j_ac], players[_pid].valid_action())
                if _ac == 0:
                    players[_pid].set_ammo(_ammo)
                joint_action[_pid] = _ac
            done, r, n_state = world.step(joint_action)
            for _pid in world.alive_players:
                players[_pid].exp_buffer.append([str(_pid) + ''.join(map(str, ob)) +
                                                 str(players[_pid].ammo) + ''.join(map(str, prev_j_ac)), joint_action[_pid], r[_pid]])
            for _pid in world.dead_in_this_step:
                players[_pid].exp_buffer.append([str(_pid) + ''.join(map(str, ob)) +
                                                 str(players[_pid].ammo) + ''.join(map(str, prev_j_ac)), joint_action[_pid], r[_pid]])
            # for _pid in world.dead_players:
            #     if r[_pid]
            prev_j_ac = copy(joint_action)
            ob = n_state
        for _pid in range(_num_players):  # TODO : replace with player 1
            v = 0
            for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                players[_pid].exp_buffer[_k][2] += _gamma * v
                v = players[_pid].exp_buffer[_k][2]
            # if v < 0:
            #     print(players[_pid].exp_buffer)
            # print(players[_pid].exp_buffer)
            sampled_exp[_pid].extend(players[_pid].exp_buffer)  # all agents share it if sp is used
        world.reset()
    q.put(sampled_exp)


def update_pi(player, exp):
    player.update_policy(exp)


def train():
    world = GridRoom()
    players = [RMAgent(i) for i in range(0, _num_players)]
    begin = time()
    # sampled_exp = []
    _save_fre = _train_iter / 10
    for _iteration in range(_train_iter):
        iter_time = time()
        sampled_exp = [[]]*_num_players
        _queue_list = []
        _exp = []
        for _th in range(_s_th):
            _q = Queue()
            p = Process(target=simulation, args=(copy(players), _q,))
            p.start()
            # p.join()
            _queue_list.append(_q)
            # _exp.append(_q.get())
        for _q in _queue_list:
            _exp.append(_q.get())
        print('Time for simulation', time()-iter_time)
        for _tn in range(_s_th):
            for _pid in range(_num_players):
                sampled_exp[_pid].extend(_exp[_th][_pid])
        print(len(sampled_exp[0]), len(sampled_exp[1]))
        # sampled_exp = simulation(players)
        # sampled_exp = [[]]*_num_players
        # for _pid in range(1, _num_players):
        #     players[_pid].u_s = copy(players[0].u_s)
        #     players[_pid].u_sa = copy(players[0].u_sa)
        #     players[_pid].average_strategy = copy(players[0].average_strategy)
        # for _i in range(_sample_iter):
        #     for _pid in range(_num_players):
        #         players[_pid].set_ammo(_ammo)
        #         players[_pid].exp_buffer = []
        #     ob = world.cur_state()
        #     done = False
        #     prev_j_ac = [9] * _num_players
        #     while not done:
        #         joint_action = [9] * _num_players
        #         for _pid in world.alive_players:
        #             _ac = players[_pid].action([ob, prev_j_ac], players[_pid].valid_action())
        #             if _ac == 0:
        #                 players[_pid].set_ammo(_ammo)
        #             joint_action[_pid] = _ac
        #         done, r, n_state = world.step(joint_action)
        #         for _pid in world.alive_players:
        #             players[_pid].exp_buffer.append([str(_pid) + ''.join(map(str, ob)) +
        #                                              str(players[_pid].ammo) + ''.join(map(str, prev_j_ac)), joint_action[_pid], r[_pid]])
        #         for _pid in world.dead_in_this_step:
        #             players[_pid].exp_buffer.append([str(_pid) + ''.join(map(str, ob)) +
        #                                              str(players[_pid].ammo) + ''.join(map(str, prev_j_ac)), joint_action[_pid], r[_pid]])
        #         # for _pid in world.dead_players:
        #         #     if r[_pid]
        #         prev_j_ac = copy(joint_action)
        #         ob = n_state
        #     for _pid in range(_num_players):  # TODO : replace with player 1
        #         v = 0
        #         for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
        #             players[_pid].exp_buffer[_k][2] += _gamma * v
        #             v = players[_pid].exp_buffer[_k][2]
        #         # if v < 0:
        #         #     print(players[_pid].exp_buffer)
        #         # print(players[_pid].exp_buffer)
        #         sampled_exp[_pid].extend(players[_pid].exp_buffer)  # all agents share it if sp is used
        #     world.reset()
        print('Episode time: %.2f' % (time() - iter_time))
        # end sample
        update_time = time()
        update_list = []
        for _pid in range(_num_players):
            p = Process(target=update_pi, args=(copy(players[_pid]), sampled_exp[_pid]))
            p.start()
            update_list.append(p)
            # players[_pid].update_policy(sampled_exp[_pid])
        for _p in update_list:
            _p.join()
        print('Update time: %.2f' %(time() - update_time))
        if (_iteration + 1) % 100 == 0:
            print('This is %d step, %.3f' % (_iteration, _iteration/100000.0))
        if (_iteration + 1) % _save_fre == 0:
            for _pid in range(_num_players):
                with open('v{}_{}.pkl'.format(_pid, _iteration / _save_fre), 'wb') as f:
                    cPickle.dump(players[0].u_s, f, 2)
                with open('q{}_{}.pkl'.format(_pid, _iteration / _save_fre), 'wb') as f:
                    cPickle.dump(players[0].u_sa, f, 2)
                with open('pi{}_{}.pkl'.format(_pid, _iteration / _save_fre), 'wb') as f:
                    cPickle.dump(players[0].average_strategy, f, 2)
    print('Time eplapsed: %.2f' % (time() - begin))


def test():
    world = GridRoom()
    players = [RMAgent(0)]
    for i in range(1, _num_players):
        players.append(RandomAgent(i))
    players[0].test = True
    begin = time()
    with open('v0.pkl', 'rb') as f:
        players[0].u_s = cPickle.load(f)
    with open('q0.pkl', 'rb') as f:
        players[0].u_sa = cPickle.load(f)
    with open('pi0.pkl', 'rb') as f:
        players[0].average_strategy = cPickle.load(f)
    total_r = np.zeros(_num_players)
    print('Time for load model: ', time()-begin, players[0].u_s.__len__(), players[0].u_sa.__len__(), players[0].average_strategy.__len__())
    begin = time()
    # sampled_exp = []
    for _iteration in range(_test_iter):
        iter_time = time()
        for _i in range(1):
            for _pid in range(_num_players):
                players[_pid].set_ammo(_ammo)
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
