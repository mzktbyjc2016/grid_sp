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
import random
from copy import *
from agent import *
import cPickle
from multiprocessing import Process, Lock, Queue, Pool
import argparse, os


args = None
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


def simulation(players, q, _seed, cur_iter):
    world = GridRoom(_seed)
    sampled_exp = []
    for _p in range(_num_players):
        sampled_exp.append([])
    for _i in range(args.sample_iter):
        for _pid in range(_num_players):
            if random.random() < 20.0/cur_iter:
                players[_pid].exploration = True
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
        world.reset()
        for _pid in range(_num_players):  # TODO : replace with player 1
            v = 0
            for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                players[_pid].exp_buffer[_k][2] += _gamma * v
                v = players[_pid].exp_buffer[_k][2]
            # if v < 0:
            #     print(players[_pid].exp_buffer)
            # print(players[_pid].exp_buffer)
            sampled_exp[_pid] += players[_pid].exp_buffer  # all agents share it if sp is used
    q.put(sampled_exp)


def update_pi(player, exp, q):
    player.update_policy(exp)
    q.put([player.u_s, player.u_sa, player.average_strategy])


def train():
    # world = GridRoom()
    players = [RMAgent(i) for i in range(_num_players)]
    for _i in range(0):
        with open('v{}_3.999.pkl'.format(_i), 'rb') as f:
            players[_i].u_s = cPickle.load(f)
        with open('q{}_3.999.pkl'.format(_i), 'rb') as f:
            players[_i].u_sa = cPickle.load(f)
        with open('pi{}_3.999.pkl'.format(_i), 'rb') as f:
            players[_i].average_strategy = cPickle.load(f)
    begin = time()
    _s_th = args.thread
    # sampled_exp = []
    _save_fre = _train_iter / 10
    for _iteration in range(_train_iter):
        iter_time = time()
        sampled_exp = []
        for _p in range(_num_players):
            sampled_exp.append([])
        _queue_list = []
        _p_list = []
        _exp = []
        for _th in range(_s_th):
            _q = Queue()
            p = Process(target=simulation, args=(copy(players), _q, None))
            p.start()
            # p.join()
            _queue_list.append(_q)
            _p_list.append(p)
            # _exp.append(_q.get())
        for _i_p, _q in enumerate(_queue_list):
            exp_from_th = _q.get()
            # print('exp_from', len(exp_from_th[0]), len(exp_from_th[1]), len(exp_from_th[2]))
            _exp.append(exp_from_th)
            # _p_list[_i_p].join()
        print(len(_exp[0][0]), len(_exp[1][0]))
        print('Time for simulation', time()-iter_time)
        for _th in range(_s_th):
            for _pid in range(_num_players):
                sampled_exp[_pid].extend(_exp[_th][_pid])
        # print(len(sampled_exp[0]), len(sampled_exp[1]), len(sampled_exp[2]))
        print('Episode time: %.2f' % (time() - iter_time))
        # end sample
        update_time = time()
        update_list = []
        update_q_list = []
        for _pid in range(_num_players):
            _q = Queue()
            p = Process(target=update_pi, args=(copy(players[_pid]), sampled_exp[_pid], _q))
            p.start()
            update_list.append(p)
            update_q_list.append(_q)
            # players[_pid].update_policy(sampled_exp[_pid])
        for _i_p, _p in enumerate(update_list):
            _updated = update_q_list[_i_p].get()
            players[_i_p].u_s = _updated[0]
            players[_i_p].u_sa = _updated[1]
            players[_i_p].average_strategy = _updated[2]
            _p.join()
        print('Update time: %.2f' %(time() - update_time))
        print('state has seen', len(players[0].u_s), len(players[1].u_s))
        for _pid in range(_num_players):
            players[_pid]._iter += 1
        if (_iteration + 1) % 100 == 0:
            print('This is %d step, %.3f' % (_iteration, _iteration/_train_iter))
        if (_iteration + 1) % _save_fre == 0:
            for _pid in range(_num_players):
                with open('v{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[0].u_s, f, 2)
                with open('q{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[0].u_sa, f, 2)
                with open('pi{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[0].average_strategy, f, 2)
    print('Time eplapsed: %.2f' % (time() - begin))


def test():
    world = GridRoom()
    players = [RMAgent(i) for i in range(3)]
    for _k in range(1, 11):
        players[0].seen = 0
        players[0].unseen = 0
        for _i in range(_num_players-2):
            with open('v{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].u_s = cPickle.load(f)
            with open('q{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].u_sa = cPickle.load(f)
            with open('pi{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].average_strategy = cPickle.load(f)
        # for i in range(1, _num_players+1):
        #     players.append(RandomAgent(i))
            players[_i].test = True
        
        begin = time()
        # with open('v0.pkl', 'rb') as f:
        #     players[0].u_s = cPickle.load(f)
        # with open('q0.pkl', 'rb') as f:
        #     players[0].u_sa = cPickle.load(f)
        # with open('pi0.pkl', 'rb') as f:
        #     players[0].average_strategy = cPickle.load(f)
        total_r = np.zeros(_num_players)
        print('Time for load model: ', time()-begin, players[0].u_s.__len__(), players[0].u_sa.__len__(), players[0].average_strategy.__len__())
        begin = time()
        # sampled_exp = []
        step = 0.0
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
                step += world.time_step
                world.reset()
            # print('Episode time: %.2f' % (time() - begin))
        print('Time eplapsed: %.2f min' % ((time() - begin)/60.0))
        print(total_r)
        print('unseen, seen, average step', players[0].unseen, players[0].seen, step/_test_iter)
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
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-si', '--sample_iter', dest='sample_iter', default=100, type=int)
    parser.add_argument('-t', '--thread', dest='thread', default=1, type=int, help='Number of thread to simulation')
    args = parser.parse_args()
    # print(args.thread, args.sample_iter)
    train()
    # test()
