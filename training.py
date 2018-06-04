#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 6/4/18 11:35 AM
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


def train():
    pass


train()
