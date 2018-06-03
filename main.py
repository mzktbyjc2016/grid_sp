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
    _sample_iter = int(config.get('algorithm', 'sample_iter'))
    _test_iter = int(config.get('algorithm', 'test_iter'))
    # _s_th = int(config.get('algorithm', 'simulation_thread'))
    _frame_stack = int(config.get('environ', 'Frame_stack'))


def one_hot(_index, dim):
    _tmp = [0]*dim
    _tmp[_index] = 1
    return _tmp


# def simulation(players_weights, _seed, cur_iter):
#     players = [NRMAgent(i) for i in range(_num_players)]
#     for i in range(_num_players):
#         players[i].update_weights(players_weights[i])
#         # players[i].update_target_weights(players[i].get_weights())
#     world = GridRoom(_seed)
#     sampled_exp = []
#     for _p in range(_num_players):
#         sampled_exp.append([])
#     for _i in range(args.sample_iter):
#         for _pid in range(_num_players):
#             if cur_iter < 2:
#                 players[_pid].exploration = True
#             players[_pid].set_ammo(_ammo)
#             players[_pid].exp_buffer = []
#         ob = world.cur_state()
#         done = False
#         prev_j_ac = [9] * _num_players
#         while not done:
#             joint_action = [9] * _num_players
#             for _pid in world.alive_players:
#                 _ac = players[_pid].action([ob, prev_j_ac], players[_pid].valid_action())
#                 joint_action[_pid] = _ac
#             done, r, n_state = world.step(joint_action)
#             for _pid in world.alive_players:
#                 players[_pid].exp_buffer.append([[_pid, ob, players[_pid].ammo, prev_j_ac], joint_action[_pid], r[_pid]])
#             for _pid in world.dead_in_this_step:
#                 players[_pid].exp_buffer.append([[_pid, ob, players[_pid].ammo, prev_j_ac], joint_action[_pid], r[_pid]])
#             for _pid, _ac in enumerate(joint_action):
#                 if _ac == 0:
#                     players[_pid].set_ammo(players[_pid].ammo-1)
#             prev_j_ac = copy(joint_action)
#             ob = n_state
#         world.reset()
#         for _pid in range(_num_players):  # TODO : replace with player 1
#             v = 0
#             for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
#                 players[_pid].exp_buffer[_k][2] += _gamma * v
#                 v = players[_pid].exp_buffer[_k][2]
#             # if v < 0:
#             #     print(players[_pid].exp_buffer)
#             # print(players[_pid].exp_buffer)
#             sampled_exp[_pid] += players[_pid].exp_buffer  # all agents share it if sp is used
#     # print('wtk')
#     print('len of exp', len(sampled_exp[0]), len(sampled_exp[1]))
#     return sampled_exp


def update_pi(player, exp, q):
    player.update_policy(exp)
    # q.put([player.u_s, player.u_sa, player.average_strategy])


def train():
    # world = GridRoom()
    players = [NRMAgent(i) for i in range(_num_players)]
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
    _tmp_weights = []
    _tmp_iter = []
    for _iteration in range(_train_iter):
        iter_time = time()
        sampled_exp = []
        players = [NRMAgent(i) for i in range(_num_players)]
        for _p in range(_num_players):
            sampled_exp.append([])
            if len(_tmp_weights) > 0:
                players[_p].update_weights(_tmp_weights[_p])
                players[_p].update_target_weights(_tmp_weights[_p])
                players[_p]._iter = _tmp_iter[_p]
        _queue_list = []
        _p_list = []
        # sampled_exp = []
        for _pid in range(_num_players):
            _tmp_weights.append(players[_pid].get_weights())
        # sampled_exp = simulation(_tmp_weights, None, _iteration+1)
        thread_list = []
        ret_code = []
        _exp = []
        for _th in range(_s_th):
            thread_list.append(subprocess.Popen(shlex.split('python simulation.py -th {} -si {} -cur_it {}'.format(_th, args.sample_iter, _iteration+1)), stdout=open('res', 'wb+'), stderr=subprocess.STDOUT))
        for _i_th in range(len(thread_list)):
            ret = thread_list[_i_th].wait(timeout=500)
            ret_code.append(ret)
        for _i_th in range(_s_th):
            with open('episodes/{}.pkl'.format(_i_th), 'rb') as ef:
                _exp.append(cPickle.load(ef))
        print('Time for simulation', time()-iter_time)
        for _th in range(_s_th):
            for _pid in range(_num_players):
                sampled_exp[_pid].extend(_exp[_th][_pid])
        # print(len(sampled_exp[0]), len(sampled_exp[1]), len(sampled_exp[2]))
        print('Episode time: %.2f' % (time() - iter_time))
        # end sample
        _tmp_weights = []
        _tmp_iter = []
        update_time = time()
        for _pid in range(_num_players):
            _t_w = players[_pid].update_policy(sampled_exp[_pid])
            # _t_w = np.load('weights_{}.npy'.format(_pid))
            _tmp_weights.append(_t_w)
            _tmp_iter.append(players[_pid]._iter)
            print(_pid, 'update done')
        # update_list = []
        # update_q_list = []
        # for _pid in range(_num_players):
        #     _q = Queue()
        #     p = Process(target=update_pi, args=(sampled_exp[_pid], _q))
        #     p.start()
        #     update_list.append(p)
        #     update_q_list.append(_q)
        #     # players[_pid].update_policy(sampled_exp[_pid])
        # for _i_p, _p in enumerate(update_list):
        #     # _updated = update_q_list[_i_p].get()
        # #     players[_i_p].u_s = _updated[0]
        # #     players[_i_p].u_sa = _updated[1]
        # #     players[_i_p].average_strategy = _updated[2]
        #     _p.join()
        print('Update time: %.2f' %(time() - update_time))
        for _pid in range(_num_players):
            players[_pid]._iter += 1
        gc.collect()
        if (_iteration + 1) % 100 == 0:
            print('This is %d step, %.3f' % (_iteration, _iteration/_train_iter))
        if (_iteration + 1) % _save_fre == 0:
            print('done')
            for _pid in range(_num_players):
                np.save('model/weights_{}_{}.npy'.format(_pid, (_iteration+1)/_save_fre), _tmp_weights[_pid])
    print('Time eplapsed: %.2f' % (time() - begin))


def test():
    world = GridRoom()
    players = [NRMAgent(i) for i in range(_num_players)]
    # players.append(RandomAgent(1))
    # players[0] = ShootingAgent(0)
    # players[1] = ShootingAgent1(1)
    for _k in range(1, 2):
        players[0].seen = 0
        players[0].unseen = 0
        players[1].seen = 0
        players[1].unseen = 0
        for _i in range(_num_players-_num_players):
            begin = time()
            with open('v{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].u_s = cPickle.load(f)
            with open('q{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].u_sa = cPickle.load(f)
            with open('pi{}_{}.0.pkl'.format(_i, _k), 'rb') as f:
                players[_i].average_strategy = cPickle.load(f)
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
        for _pid in range(_num_players):
            print('unseen, seen, average step', players[_pid].unseen, players[_pid].seen, step/_test_iter)
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
