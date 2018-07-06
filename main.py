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
# from multiprocessing import Process, Lock, Pool, Queue
import subprocess32 as subprocess
import shlex
import argparse, os, gc
from multiprocessing import cpu_count
# from simulation import simulation


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
    # _sample_iter = int(config.get('algorithm', 'sample_iter'))
    _test_iter = int(config.get('algorithm', 'test_iter'))
    # _s_th = int(config.get('algorithm', 'simulation_thread'))
    _frame_stack = int(config.get('environ', 'Frame_stack'))
    _max_epi = int(config.get('algorithm', 'max_num_episodes'))
    _trunc_prob = float(config.get('algorithm', 'truncated_prob'))


def one_hot(_index, dim):
    _tmp = [0]*dim
    _tmp[_index] = 1
    return _tmp


def update_pi(player, exp, q):
    player.update_policy(exp)
    # q.put([player.u_s, player.u_sa, player.average_strategy])


def train():
    # world = GridRoom()
    players = [NRMAgent(i) for i in range(_num_players)]
    begin = time()
    # sampled_exp = []
    _save_fre = _train_iter / 10
    _tmp_weights = []
    total_epi = 0
    if not os.path.exists('index'):
        os.mkdir('index')
    for _iteration in range(_train_iter):
        iter_time = time()
        _s_th = min(args.thread, cpu_count())
        _sample_iter = max(int(_max_epi * 0.1 * max(_trunc_prob, 1.0/max(_iteration-9, 1)) / (_s_th * _num_players)), 1)
        # players = [NRMAgent(i) for i in range(_num_players)]
        for _p in range(_num_players):  # share weights
            if _p == 0:
                if len(_tmp_weights) > 0:
                    players[_p].update_weights(_tmp_weights[_p])
                    players[_p].update_target_weights(_tmp_weights[_p])
            else:
                players[_p].update_weights(players[0].get_weights())
                players[_p].update_target_weights(players[0].get_weights())

        for _pid in range(1):
            _tmp_weights.append(players[_pid].get_weights())
        # sampled_exp = simulation(_tmp_weights, None, _iteration+1)
        thread_list = []
        ret_code = []
        ir_list = []
        if _sample_iter*_num_players*_s_th + total_epi <= _max_epi:  # assign the corresponding TFRecord index and each simulation thread produces sample_iter*num_players episodes
            ir_list = np.array(range(total_epi, total_epi+_sample_iter*_num_players*_s_th), dtype=np.int64)
            for _ in range(_s_th):
                np.save('index/{}.npy'.format(_), ir_list[_sample_iter*_num_players*_: _sample_iter*_num_players*(_+1)])
        elif total_epi < _max_epi:
            _temp = range(total_epi, _max_epi)
            _temp.extend(sample(range(total_epi), _s_th*_sample_iter*_num_players + total_epi - _max_epi))
            ir_list = np.array(_temp, dtype=np.int64)
            for _ in range(_s_th):
                np.save('index/{}.npy'.format(_), ir_list[_sample_iter*_num_players*_: _sample_iter*_num_players*(_+1)])
        else:  # total episodes exceed the max episodes buffer size
            ir_list = np.array(sample(range(_max_epi), _s_th * _sample_iter*_num_players), dtype=np.int64)
            for _ in range(_s_th):
                np.save('index/{}.npy'.format(_), ir_list[_sample_iter*_num_players*_: _sample_iter*_num_players*(_+1)])

        for _th in range(_s_th):
            thread_list.append(subprocess.Popen(shlex.split('python simulation.py -th {} -si {} -cur_it {}'.format(_th, _sample_iter, _iteration+1)), stdout=open('res', 'wb+'), stderr=subprocess.STDOUT))
        for _i_th in range(len(thread_list)):
            ret = thread_list[_i_th].wait()
            ret_code.append(ret)
        # for _i_th in range(_s_th):
        #     with open('episodes/{}.pkl'.format(_i_th), 'rb') as ef:
        #         _exp.append(cPickle.load(ef))
        # print('Time for simulation', time()-iter_time)
        total_epi += _s_th*_sample_iter*_num_players
        print('Total episodes: ', total_epi)
        # print(len(sampled_exp[0]), len(sampled_exp[1]), len(sampled_exp[2]))
        print('Episode simulation time: %.2f' % (time() - iter_time))
        # end sample
        _tmp_weights = []
        _tmp_iter = []
        update_time = time()
        for _pid in range(1):
            print(players[_pid].get_weights()[2][1][0:4])
            _t_w = players[_pid].update_policy(total_epi)
            # _t_w = np.load('weights_{}.npy'.format(_pid))
            _tmp_weights.append(_t_w)
            _tmp_iter.append(players[_pid]._iter)
        print('Update time: %.2f' %(time() - update_time))
        for _pid in range(_num_players):
            players[_pid]._iter += 1
        gc.collect()
        if (_iteration + 1) % _save_fre == 0:
            for _pid in range(1):
                np.save('model/weights_{}_{}.npy'.format(_pid, int((_iteration+1)/_save_fre)), _tmp_weights[_pid])
    print('Time eplapsed: %.2f' % (time() - begin))


def test():
    world = GridRoom()
    players = [RandomAgent(i) for i in range(_num_players)]
    players[0] = NRMAgent(0)
    # players[0] = NRMAgent(0)
    # players[0] = NRMAgent(0)
    # players.append(RandomAgent(1))
    # players[0] = ShootingAgent(0)
    # players[1] = ShootingAgent1(1)
    for _k in range(1, 11):
        players[0].seen = 0
        players[0].unseen = 0
        players[1].seen = 0
        players[1].unseen = 0
        for _i in range(_num_players-2):
            begin = time()
            players[_i].update_weights(np.load('model/weights_{}_{}.0.npy'.format(_i, _k)))
            players[_i].test = False
            players[_i].exploration = True
            # print('Time for load model: ', time() - begin, players[_i].u_s.__len__(), players[_i].u_sa.__len__(), players[_i].average_strategy.__len__())
        players[1].test = False
        players[1].exploration = False
        total_r = np.zeros(_num_players)
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
                        joint_action[_pid] = _ac
                    done, r, n_state = world.step(joint_action)
                    # if done:
                    #     print(world.dead_in_this_step, world.time_step, ob, prev_j_ac, players[0].ammo, n_state, joint_action, players[1].ammo)
                    prev_j_ac = copy(joint_action)
                    ob = n_state
                    for _pid, _ac in enumerate(joint_action):
                        if _ac == 0:
                            players[_pid].set_ammo(players[_pid].ammo - 1)
                total_r = np.add(total_r, world.players_total_reward)
                step += world.time_step
                world.reset()
            # print('Episode time: %.2f' % (time() - begin))
        print('Time eplapsed: %.2f min' % ((time() - begin)/60.0))
        print(total_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-si', '--sample_iter', dest='sample_iter', default=100, type=int)
    parser.add_argument('-t', '--thread', dest='thread', default=1, type=int, help='Number of thread to simulation')
    args = parser.parse_args()
    # print(args.thread, args.sample_iter)
    if not os.path.exists('episodes'):
        os.mkdir('episodes')
    if not os.path.exists('model'):
        os.mkdir('model')
    train()
    test()
    # players = [RMAgent(i) for i in range(_num_players)]
    # with open('pi{}_{}.0.pkl'.format(0, 10), 'rb') as f:
    #     players[0].average_strategy = cPickle.load(f)
    # for i in players[0].average_strategy:
    #     print(i, players[0].average_strategy[i])
