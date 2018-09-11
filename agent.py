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
from time import *
import logging
import os


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
gpu_options = tf.GPUOptions(allow_growth=True)
gpu_config = tf.ConfigProto(gpu_options=gpu_options)
#os.environ['CUDA_VISIBLE_DEVICES']=''
config = ConfigParser.ConfigParser()
with open('config.cfg', 'rw') as cfgfile:
    config.readfp(cfgfile)
    _width = int(config.get('environ', 'Width'))
    _height = int(config.get('environ', 'Height'))
    _num_players = int(config.get('environ', 'Players'))
    _ammo = int(config.get('environ', 'Ammo'))
    _max_step = int(config.get('environ', 'Max_Step'))
    _gamma = float(config.get('environ', 'Gamma'))
    _frame_stack = int(config.get('environ', 'Frame_stack'))
    _train_iter = int(config.get('algorithm', 'train_iter'))
    _sample_iter = int(config.get('algorithm', 'sample_iter'))
    _test_iter = int(config.get('algorithm', 'test_iter'))
    # _s_th = int(config.get('algorithm', 'simulation_thread'))
    _batch_size = int(config.get('algorithm', 'batch_size'))
    _num_gpu = int(config.get('algorithm', 'num_gpu'))
    _num_epochs = int(config.get('algorithm', 'num_epochs'))


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
        one_hot_state = []
        # print(state[0], state[1])
        for i in range(len(state[0])):  # state[0] is a 3*N int list where each 3-tuple consists of (x, y, dir)
            if i % 3 == 0:
                one_hot_state += one_hot(state[0][i], _width)
            elif i % 3 == 1:
                one_hot_state += one_hot(state[0][i], _height)
            else:
                one_hot_state += one_hot(state[0][i] - 1, 4)
        for i in range(len(state[1])):
            if state[1][i] != 9:
                one_hot_state += one_hot(state[1][i], 5)
            else:
                one_hot_state += [0] * 5
        for _, _item in enumerate(state[0][self.id*3: self.id*3+3]):  # duplicate the position and direction information to identify each player and thus the state can be reused (shared by all the players)
            if _ == 0:
                one_hot_state = one_hot(_item, _width) + one_hot_state
            elif _ == 1:
                one_hot_state = one_hot(_item, _height) + one_hot_state
            else:
                one_hot_state = one_hot(_item-1, 4) + one_hot_state
        return one_hot_state

    def action(self, state, action_space):
        # return choice(action_space)
        if not self.test:
            state_key = str(self.id) + ''.join(map(str, state[0])) + str(self.ammo) + ''.join(map(str, state[1]))
            if self.exploration:
                return choice(action_space)
            if state_key not in self.u_s:
                self.unseen += 1
                self.wtk.add(state_key)
                return choice(action_space)
            else:
                _imm_regret = []
                var_u_s = self.u_s[state_key]
                exp_u = var_u_s[0] / var_u_s[1]
                for _ac in action_space:
                    s_a_key = state_key + str(_ac)
                    if s_a_key not in self.u_sa:
                        exp_u_a = 0.0
                    else:
                        var_u_sa = self.u_sa[s_a_key]
                        exp_u_a = var_u_sa[0] / var_u_sa[1]
                    _imm_regret.append(exp_u_a - exp_u)
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
                    # if state_key not in self.wtk and prob[0] > 0.2:
                    #     with open('{}.prob'.format(self.id), 'ab+') as f:
                    #         f.write(state_key + ' ' + str(prob) + '\n')
                    #     self.wtk.add(state_key)
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
                ac_v = [0] * 5
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
        if (pos1[0] - pos2[0]) * (pos1[1] - pos2[1]) != 0:  # not in the same horizontal or vertical line
            return False
        else:
            if pos1[0] == pos2[0]:  # in the same row and will return False when pos1 coincides with pos2
                if dir1 == 3 or dir1 == 4:
                    return False
                return np.dot((0, pos2[1] - pos1[1]), (0, (-1) ** dir1)) > 0
            else:  # in the same column
                if dir1 == 1 or dir1 == 2:
                    return False
                return np.dot((pos2[0] - pos1[0], 0), ((-1) ** dir1, 0)) > 0


def one_hot(_index, dim):
    _tmp = [0] * dim
    _tmp[_index] = 1
    return _tmp


def decode_from_tfrecords(filename_queue, batch_size=1):
    reader = tf.TFRecordReader()
    _, queue_batch = reader.read_up_to(filename_queue, batch_size)
    if batch_size > 0:
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        batch_serialized_example = tf.train.shuffle_batch([queue_batch],
                                                          batch_size=batch_size,
                                                          num_threads=3,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue,
                                                          enqueue_many=True)

    features = tf.parse_example(batch_serialized_example,
                                features={
                                    'State': tf.FixedLenFeature([(_height + _width + 4) + (9 + _height + _width) * _num_players + 3], tf.float32),
                                    'Return': tf.FixedLenFeature([1], tf.float32),
                                    'Act': tf.FixedLenFeature([1], tf.float32),
                                    'Act_prob': tf.FixedLenFeature([1], tf.float32)
                                })
    state = features['State']
    cumulated_reward = features['Return']
    act = features['Act']
    act_prob = features['Act_prob']

    return state, cumulated_reward, act, act_prob


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.


    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def is_gpu_available(cuda_only=True):
    """
    code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    Returns whether TensorFlow can access a GPU.
    Args:
      cuda_only: limit the search to CUDA gpus.
    Returns:
      True iff a gpu device of the requested kind is available.
    """
    from tensorflow.python.client import device_lib as _device_lib

    if cuda_only:
        return any((x.device_type == 'GPU')
                   for x in _device_lib.list_local_devices())
    else:
        return any((x.device_type == 'GPU' or x.device_type == 'SYCL')
                   for x in _device_lib.list_local_devices())


class NRMAgent(RMAgent):
    """
    This is a neural agent that uses mlp only and the input state is encoded as follow
        ID(1)+ (one hot pos (N+M) + one hots dir (4))* # of players+ ammo(1, [0,1] float)*# of repetition + one hot prev_joint_action (# of players*5)
        (10+N+M)* # of players + repetition
    """

    def __init__(self, player_id=0, simulation=False):
        super(NRMAgent, self).__init__(player_id)
        # TODO overload policy and value function
        if not simulation:
            os.environ['CUDA_VISIBLE_DEVICES']='0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=''
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=gpu_config)
        model = Model(self.graph)
        self.out, self.t_out, self.h_state, self.get_weights, self.update_weights, self.average_strategy, self.behavior_pi, self.update_target_weights, self.get_cu_re = \
            model.create_model(self.session, input_dim=(_height + _width + 4) + (9 + _height + _width) * _num_players + 3, out_dim=11, action_dim=5)
        # self.out2, self.update_weights = model.create_target_model(input_dim=(10 + _height + _width) * _num_players + 3, out_dim=11, action_dim=5)
        # self.session.run(tf.local_variables_initializer())
        # self.session.run(tf.global_variables_initializer())
        # tf.reset_default_graph()
        # with self.graph.as_default():
        #     weights = self.session.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pi'))
        weights = self.get_weights()
        # for w in weights:
        #     print(np.shape(w))
        self.update_target_weights(weights)
        self._iter = 1
        self.history_buffer = []

    def action(self, state, action_space):
        one_hot_state = self.parse_state(state)
        s = one_hot_state + [self.ammo / float(_ammo)] * 3
        if not self.test:
            prob = self.behavior_pi(s)
        else:
            prob = self.average_strategy(s)
        if len(action_space) > 4:
            return np.random.choice(action_space, p=prob)
        else:
            if sum(prob[1:]) > 0:
                trunc_prob = np.true_divide(prob[1:], sum(prob[1:]))
            else:
                trunc_prob = np.ones(len(action_space))/len(action_space)
            return np.random.choice(action_space, p=trunc_prob)

    def update_policy(self, num_tfrecords=1):
        graph = tf.Graph()
        session = tf.Session(graph=graph, config=gpu_config)
        batch_size = _batch_size
        n_epochs = _num_epochs

        lr = 1e-1
        num_gpus = _num_gpu
        logdir = 'save'+'/'+str(self.id)
        c1 = 0
        c2 = 0.5
        c3 = 0.1

        os.environ['CUDA_VISIBLE_DEVICES']='0'
        no_gpu = not is_gpu_available()
        if no_gpu:
            num_gpus = 1
        with session as sess:

            tower_grads = []
            tower_vars = []
            tf_record_list = ['episodes/{}/{}.tfrecords'.format(self.id, i) for i in xrange(num_tfrecords)]
            # print(tf_record_list)
            filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=n_epochs, shuffle=True)
            state, cu_re, label, act_prob = decode_from_tfrecords(filename_queue, batch_size=batch_size * num_gpus)
            act = tf.cast(label, tf.int32)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            global_iteration = tf.Variable(1, name='global_iteration', trainable=False)
            train_explosion = False
            lr = tf.train.exponential_decay(lr, global_step, 50, 0.9925, staircase=True)
            add_global = global_step.assign_add(1)
            opt = tf.train.GradientDescentOptimizer(lr)

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(num_gpus):
                    with tf.device('/cpu:0' if no_gpu else '/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                            h_state = tf.placeholder(tf.float32, [None, (_height + _width + 4) + (9 + _height + _width) * _num_players + 3], name='h_state')
                            h_act = tf.placeholder(tf.int32, [None, 1], name='target_action{}'.format(i))
                            # h_act_prob = tf.placeholder(tf.float32, [None, 5], name='target_action_prob{}'.format(i))
                            h_cu_re = tf.placeholder(tf.float32, [None, 1], name='cumulated_reward{}'.format(i))  # return from current state
                            h_cu_re_a = tf.placeholder(tf.float32, [None, 5], name='cumulated_reward_a{}'.format(i))  # return from current after taking some action

                            # pi_weight = tf.placeholder(tf.float32, [None, 1], name='pi_weight{}'.format(i))
                            # q_weight = tf.placeholder(tf.float32, [None, 1], name='q_weight{}'.format(i))
                            # v_weight = tf.placeholder(tf.float32, [None, 1], name='v_weight{}'.format(i))
                            q_weight = 1.0
                            v_weight = 1.0

                            out, t_out, h_input, update_params, update_target_params, get_weights = build_training_model(h_state, 11, 5)
                            # qv_value = out[:, 5: 11]
                            cross_entropy_mean = tf.nn.softmax_cross_entropy_with_logits(logits=out[:, 0:5],
                                                                                         labels=tf.one_hot(h_act, 5))
                            # pi_loss = tf.reduce_mean(cross_entropy_mean)

                            q_error = tf.reduce_mean(tf.abs(out[:, 5:10]-h_cu_re_a), axis=1)
                            v_error = tf.reduce_mean(tf.abs(out[:, 10:11]-h_cu_re), axis=1)
                            # pi_error = tf.abs(out[:, 0:5])
                            pi_loss = cross_entropy_mean
                            # pi_loss = tf.losses.compute_weighted_loss(cross_entropy_mean, weights=pi_weight, reduction=tf.losses.Reduction.NONE)

                            if tf.__version__ >= '2.3':  # just use mse
                                q_loss = tf.losses.huber_loss(predictions=out[:, 5:10], labels=h_cu_re_a, delta=1.0, weights=q_weight, reduction=tf.losses.Reduction.NONE)
                                v_loss = tf.losses.huber_loss(predictions=out[:, 10:11], labels=h_cu_re, delta=1.0, weights=v_weight, reduction=tf.losses.Reduction.NONE)
                            else:
                                q_loss = tf.losses.mean_squared_error(predictions=out[:, 5:10], labels=h_cu_re_a, weights=q_weight, reduction=tf.losses.Reduction.NONE)
                                v_loss = tf.losses.mean_squared_error(predictions=out[:, 10:11], labels=h_cu_re, weights=v_weight, reduction=tf.losses.Reduction.NONE)

                            q_loss = tf.reduce_mean(q_loss, axis=1)
                            v_loss = tf.reduce_mean(v_loss, axis=1)

                            # pi_logits = tf.nn.softmax(logits=self.out[:, 0:6])
                            # entropy = -tf.reduce_mean(tf.reduce_sum(-tf.log(pi_logits) * pi_logits, axis=1))  # negative of entropy term so that can be directly added in the loss function which is minimized
                            loss = tf.reduce_mean(c1 * v_loss + c2 * q_loss + c3 * pi_loss)
                            tower_vars.append(([h_state, h_cu_re, h_act, h_cu_re_a, q_weight, v_weight], q_loss,
                                               v_loss, pi_loss, loss))
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            train_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "pi")
                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads, _vars = zip(*opt.compute_gradients(loss, var_list=train_weights))
                            grads, _ = tf.clip_by_global_norm(grads, batch_size)
                            # Keep track of the gradients across all towers.
                            tower_grads.append(zip(grads, _vars))

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = average_gradients(tower_grads)

            tower_holders, tower_q_losses, tower_v_losses, tower_pi_loss, tower_losses = zip(*tower_vars)
            summaries.append(tf.summary.scalar('learning_rate', lr))
            summaries.append(tf.summary.scalar('q loss', tf.reduce_mean(tower_q_losses)))
            summaries.append(tf.summary.scalar('value loss', tf.reduce_mean(tower_v_losses)))
            summaries.append(tf.summary.scalar('policy loss', tf.reduce_mean(tower_pi_loss)))
            summaries.append(tf.summary.scalar('total weighted loss', tf.reduce_mean(tower_losses)))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Group all updates to into a single train op.
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # to ensure testing phase works properly
            # with tf.control_dependencies([add_global]):
            train_op = apply_gradient_op

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)
            writer = tf.summary.FileWriter(logdir, graph)

        # with self.session as sess, sess.as_default():
        #     print(self.get_weights()[2][1][0:4])
            update_params(sess, self.get_weights())
            update_target_params(sess, self.get_weights())
            begin = time()
            step = 0
            if not os.path.exists('global_step{}'.format(self.id)):
                shift_step = 0
            else:
                with open('global_step{}'.format(self.id), 'rb') as gs_f:
                    shift_step = int(gs_f.read())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())
            try:
                while not coord.should_stop():
                    # print(len(batch_data['return']))
                    start_time = time()
                    _state, _cu_re, _act = sess.run([state, cu_re, act])  # 'state', 'return', 'action'
                    _t_f = {}
                    _q_error_f = {}
                    _v_error_f = {}
                    # print(np.shape(_state))
                    _feed_qv = []
                    for _k in range(num_gpus):  # [h_state, h_cu_re, h_act, h_cu_re_a]
                        _t_f[tower_holders[_k][0]] = _state[_k * batch_size: (_k + 1) * batch_size]
                        # _q_error_f[tower_holders[_k][0]] = _state[_k * batch_size: (_k + 1) * batch_size]
                        # _feed_qv.append(tower_QVs[_k](_state[_k * batch_size: (_k + 1) * batch_size]))
                    # _feed_qv = np.squeeze(_feed_qv)
                    for _k in range(num_gpus):
                        _g = np.reshape(_cu_re[_k * batch_size: (_k + 1) * batch_size], [batch_size, 1])  # the return g_k = \sum \gamma^k r_{n+k}
                        # print(np.shape(_g))
                        _action_taken = _act[_k * batch_size: (_k + 1) * batch_size]
                        _t_f[tower_holders[_k][1]] = _g  # to fit ( T-1 iter + this iter )
                        _t_f[tower_holders[_k][2]] = _action_taken
                        _t_f[tower_holders[_k][3]] = _action_taken * _g.repeat(5, axis=1)  # np.eye(C)[index] return one_hot array of index
                    # logger.info('Time for fetch and feed dict Step %d: %.5f sec a step' % (step+1, time.time()-start_time))
                    if (step + 1) % 100 == 0:
                        # start_time = time.time()
                        _, __, summary = sess.run([train_op, add_global, summary_op], feed_dict=_t_f)
                        writer.add_summary(summary, shift_step+step)
                        duration = time() - start_time
                        if (step+1) % 500 == 0:
                            logger.info('Step %d: %.5f sec a step' % (step + 1, duration))
                    else:
                        # print(sess.run([tf.shape(tower_q_losses), tf.shape(tower_v_losses), tf.shape(tower_pi_loss)], feed_dict=_t_f))
                        _, __ = sess.run([train_op, add_global], feed_dict=_t_f)

                    step += 1
            # except tf.errors.InvalidArgumentError:
            #     train_explosion = True
            #     print('grad explosion')
            #     pass
            except tf.errors.OutOfRangeError:
                logger.info('Done training for %d epochs, %d steps.' % (n_epochs, step))
                logger.info('total time elapsed: %.2f min' % ((time() - begin) / 60.0))
                # with open('mgb_{}_{}.log'.format(num_gpus, batch_size), 'a+') as rlfile:
                #     rlfile.write('%.2f min\n' % ((time.time() - begin) / 60.0))
                t_weights = get_weights(sess)
                np.save('weights_{}.npy'.format(self.id), get_weights(sess))
                with open('traintime.log', 'a+') as runlogfile:
                    runlogfile.write('total time elapsed: %.2f min\n' % ((time() - begin) / 60.0))
                with open('global_step{}'.format(self.id), 'wb+') as global_step_f:
                    global_step_f.write(str(step+shift_step))
            finally:
                coord.request_stop()
                print('Training Done')
            coord.join(threads)
        return t_weights

    def parse_state(self, state):
        one_hot_state = []
        # print(state[0], state[1])
        for i in range(len(state[0])):  # state[0] is a 3*N int list where each 3-tuple consists of (x, y, dir)
            if i % 3 == 0:
                one_hot_state += one_hot(state[0][i], _width)
            elif i % 3 == 1:
                one_hot_state += one_hot(state[0][i], _height)
            else:
                one_hot_state += one_hot(state[0][i] - 1, 4)
        for i in range(len(state[1])):
            if state[1][i] != 9:
                one_hot_state += one_hot(state[1][i], 5)
            else:
                one_hot_state += [0] * 5
        for _, _item in enumerate(state[0][self.id*3: self.id*3+3]):  # duplicate the position and direction information to identify each player and thus the state can be reused (shared by all the players)
            if _ == 0:
                one_hot_state = one_hot(_item, _width) + one_hot_state
            elif _ == 1:
                one_hot_state = one_hot(_item, _height) + one_hot_state
            else:
                one_hot_state = one_hot(_item-1, 4) + one_hot_state
        return one_hot_state
