#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: The grid sp environment
@Date: 5/19/18 7:29 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''

from __future__ import print_function
from __future__ import division
import ConfigParser
import numpy as np
import random
from time import *
from copy import *

DIRECTION = ['←', '→', '↑', '↓']

config = ConfigParser.ConfigParser()
with open('config.cfg', 'rw') as cfgfile:
    config.readfp(cfgfile)
    _width = int(config.get('environ', 'Width'))
    _height = int(config.get('environ', 'Height'))
    _num_players = int(config.get('environ', 'Players'))
    _ammo = int(config.get('environ', 'Ammo'))
    _max_step = int(config.get('environ', 'Max_Step'))
    _gamma = float(config.get('environ', 'Gamma'))
    _kill = float(config.get('environ', 'Kill'))
    _death = float(config.get('environ', 'Death'))
    _step = float(config.get('environ', 'Step'))
    _timeout = float(config.get('environ', 'Timeout'))
    _only_alive = float(config.get('environ', 'Only_alive'))



class GridRoom(object):
    """
    Grid Room is a N*M grid world with K players. Each player is randomly spawned in the one of the grid
    and with random direction. Each player has five action and some amount of ammo. The five actions are
    fire, left, right, up, and down (indexed 0~5). When the next position is empty and moving direction
    is not facing the wall, it moves, else it stays. In the environment, we assume that each player can
    only see the enemies straight ahead and do not try to cooperate with others. If the player fires and
    the enemy is in front of him do not move to another lane, i.e the enemy is still straight ahead of
    him, he gets a score and consumes one ammo. The goal of each agent is trying to get higher score and
    avoid being killed. And after some maximum time step, the game ends. Thus with different configuration,
    the game can be viewed as an abstraction of pure FPS game like ViZDoom Death Match or
    PLAYERUNKNOWN’S BATTLEGROUNDS.
    """

    def __init__(self):
        self.state = []
        for row in range(_width):
            tmp = []
            for col in range(_height):
                tmp.append([0, 0])
            self.state.append(tmp)
        self.players_pos_dir = [[]]*_num_players
        self.fire_pos_dir = [[]]*_num_players
        init_position = random.sample([_i*(_height+1) for _i in range( _height)], _num_players)
        init_direction = np.random.randint(1, 5, _num_players)
        # init_position = [0, 3]
        # init_direction = [4, 3]
        for i in range(_num_players):
            _pos_y = init_position[i] % _width  # range in (0, width-1)
            _pos_x = init_position[i] // _height  # range in (0, height-1)
            # self.state[_pos_x][_pos_y] = [init_direction[i], i+1]
            self.players_pos_dir[i] = [[_pos_x, _pos_y], init_direction[i]]
        self.ammo = [_ammo] * _num_players
        self.dead_players = []
        self.dead_in_this_step = []
        self.alive_players = range(_num_players)
        self.players_total_reward = [0] * _num_players
        self.time_step = 0
        self.max_step = _max_step
        self.r_kill = _kill
        self.r_death = _death
        self.r_step = _step
        self.r_timeout = _timeout
        self.r_only_alive = _only_alive

    def step(self, joint_action):
        """
        :param joint_action: actions of player 1 to K
        :return: the next state after joint actions are made and the joint reward
        """
        reward = [self.r_step]*_num_players
        self.dead_in_this_step = []
        is_terminal = False
        for _p in self.dead_players:
            reward[_p] = 0
        if self.time_step >= _max_step:  # TODO: reach maximum steps
            reward = [self.r_step+self.r_timeout]*_num_players
            for _p in self.dead_players:
                reward[_p] = 0
            for _p in self.alive_players:
                self.players_total_reward[_p] += reward[_p]
            return True, reward, [0]*3*_num_players
        next_state = [0]*3*_num_players
        fired_players = [i for i, v in enumerate(joint_action) if i not in self.dead_players and v == 0]
        for i in range(_num_players):  # allow multiple players come into the same grid
            if i in self.dead_players:
                continue
            if i in fired_players:
                continue
            action = joint_action[i]
            _x = self.players_pos_dir[i][0][0]
            _y = self.players_pos_dir[i][0][1]
            if (action-1) // 2 == 0:
                _y += (-1)**action
                if 0 <= _y < _width and self.state[_x][_y][0] > -1:
                    self.players_pos_dir[i][0][1] = _y
            else:
                _x += (-1)**action
                if 0 <= _x < _height and self.state[_x][_y][0] > -1:
                    self.players_pos_dir[i][0][0] = _x
            self.players_pos_dir[i][1] = action
        for _p in self.alive_players:  # judge if someone will be killed
            face_ammo_from = []
            ammo_distance = []
            for _id, var in enumerate(self.fire_pos_dir):
                if _id != _p and len(var) > 0:
                    if self.is_in_front(var[0], var[1], self.players_pos_dir[_p][0]):
                        face_ammo_from.append(_id)
                        ammo_distance.append(np.abs(self.players_pos_dir[_p][0][0]-var[0][0]) + np.abs(self.players_pos_dir[_p][0][1]-var[0][1]))  # because in the same row or col
            if len(face_ammo_from) > 0:
                player_id = _p  # this player is killed
                _min_dis_index = np.argmin(ammo_distance)
                _id = face_ammo_from[_min_dis_index]
                self.players_pos_dir[player_id] = []
                self.fire_pos_dir[_id] = []
                self.dead_players.append(player_id)
                reward[player_id] += self.r_death
                reward[_id] += self.r_kill
                self.dead_in_this_step.append(player_id)
                # print('{} is killed by {}'.format(player_id, _id))
        self.alive_players = [k for k in range(_num_players) if k not in self.dead_players]
        if len(self.alive_players) == 1:
            is_terminal = True
            reward[self.alive_players[0]] += self.r_only_alive

        for _id in fired_players:
            if _id not in self.dead_players:
                self.fire_pos_dir[_id] = [[self.players_pos_dir[_id][0][0], self.players_pos_dir[_id][0][1]], self.players_pos_dir[_id][1]]
        for _p in self.alive_players:
            next_state[_p] = self.players_pos_dir[_p][0][0]
            next_state[_p+1] = self.players_pos_dir[_p][0][1]
            next_state[_p+2] = self.players_pos_dir[_p][1]
        for _p in range(_num_players):
            self.players_total_reward[_p] += reward[_p]
        self.time_step += 1
        return is_terminal, reward, next_state

    def reset(self):
        self.__init__()

    def cur_state(self):
        _state = [0] * 3 * _num_players
        for _p in self.alive_players:
            _state[_p] = self.players_pos_dir[_p][0][0]
            _state[_p+1] = self.players_pos_dir[_p][0][1]
            _state[_p+2] = self.players_pos_dir[_p][1]
        return _state

    def print_state(self):
        for row in self.state:
            for col in row:
                if col[0] == 0:
                    print(col[0], end=' ')
                else:
                    print(DIRECTION[col[0]-1], end=' ')
            print('\t\t', end='')
            for col in row:
                if col[0] == 0:
                    print(0, end=' ')
                else:
                    print(col[1], end=' ')
            print('')
        print('')

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
                return np.dot((0, pos2[1] - pos1[1]), (0, (-1)**dir1)) > 0
            else:  # in the same column
                return np.dot((pos2[0]-pos1[0], 0), ((-1)**dir1, 0)) > 0


if False:
    world = GridRoom()
    # print(world.players_pos_dir)
    # world.print_state()
    # jot_action = [0, 0, 0]
    # world.step(jot_action)
    # # print(world.players_pos_dir)
    # world.print_state()
    # world.step([4, 4, 4])
    # world.print_state()
    begin = time()
    for _i in range(1000):
        done = False
        while not done:
            jot_action = np.random.randint(0, 5, _num_players)
            # print(jot_action)
            done, r, n_state = world.step(jot_action)
            # world.print_state()
        # print(world.players_total_reward, n_state)
        world.reset()
    print('Time eplapsed: %.2f' % (time()-begin))
