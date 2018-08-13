#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 6/4/18 1:08 AM
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
import argparse, os, gc
import tensorflow as tf


# args = None
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

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-si', '--sample_iter', dest='sample_iter', default=100, type=int)
parser.add_argument('-th', '--thread', dest='thread', default=1, type=int, help='index of thread')
parser.add_argument('-cur_it', dest='cur_iter', default=1, type=int, help='current iteration')
args = parser.parse_args()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int_list=tf.train.Int64List(value=value))


def simulation(_seed, cur_iter):
    players = [NRMAgent(i, True) for i in range(_num_players)]
    for i in range(_num_players):
        if i == 0:
            if os.path.exists('weights_{}.npy'.format(0)):
                tmp_weights = np.load('weights_{}.npy'.format(0))
                players[i].update_weights(tmp_weights)
        else:
            players[i].update_weights(players[0].get_weights())
        # players[i].update_target_weights(players[i].get_weights())
    world = GridRoom(_seed)
    sampled_exp = []
    ir_list = np.load('index/{}.npy'.format(args.thread))
    for _p in range(_num_players):
        sampled_exp.append([])
    for _i in range(args.sample_iter):
        for _pid in range(_num_players):
            if cur_iter < 2:
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
                joint_action[_pid] = _ac
            done, r, n_state = world.step(joint_action)
            for _pid in world.alive_players:
                players[_pid].exp_buffer.append([[_pid, ob, players[_pid].ammo, prev_j_ac], joint_action[_pid], r[_pid]])
            for _pid in world.dead_in_this_step:
                players[_pid].exp_buffer.append([[_pid, ob, players[_pid].ammo, prev_j_ac], joint_action[_pid], r[_pid]])
            for _pid, _ac in enumerate(joint_action):
                if _ac == 0:
                    players[_pid].set_ammo(players[_pid].ammo-1)
            prev_j_ac = copy(joint_action)
            ob = n_state
        world.reset()
        for _pid in range(1):
            # print('write player')
            train_writer = tf.python_io.TFRecordWriter('episodes/{}.tfrecords'.format(ir_list[_i+_pid]))
            v = 0
            for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                _item = players[_pid].exp_buffer[_k]
                players[_pid].exp_buffer[_k][2] += _gamma * v
                v = players[_pid].exp_buffer[_k][2]
                one_hot_s = players[_pid].parse_state([_item[0][1], _item[0][3]])
                full_state = one_hot_s + [_item[0][2] / float(_ammo)] * 3
                b_feature = {'State': _float_feature(full_state),
                             'Return': _float_feature([v]),
                             'Act': _float_feature([_item[1]]), 'Act_prob': _float_feature([1.0])}
                example = tf.train.Example(features=tf.train.Features(feature=b_feature))
                train_writer.write(example.SerializeToString())
            train_writer.close()
            # if v < 0:
            #     print(players[_pid].exp_buffer)
            # print(players[_pid].exp_buffer)
            # sampled_exp[_pid] += players[_pid].exp_buffer  # all agents share it if sp is used
    # print('wtk')
    # print('len of exp', len(sampled_exp[0]), len(sampled_exp[1]))
    # with open('episodes/{}.pkl'.format(args.thread), 'wb') as ef:
    #     cPickle.dump(sampled_exp, ef, 2)


# print(args.thread, args.sample_iter, args.cur_iter)
begin = time()
simulation(None, args.cur_iter)
print('time elapsed: ', time() - begin)
