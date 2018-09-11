#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 5/23/18 8:56 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''
from __future__ import print_function
from __future__ import division
from time import *
from multiprocessing import Pool, Queue, Process
from random import *
import cPickle
import numpy as np
# import tensorflow as tf
from copy import copy
from multiprocessing import cpu_count
from tfmodel import *
import ConfigParser
import logging
import os, gc
import math
import tfmodel
from environment import GridRoom
from agent import *
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


def test(q):
    x = []
    for i in xrange(100000):
        x.append(choice([1, 2, 3, 4, 5]))
    q.put(x)


def copy_test(b_dict):
    return


# for i in range(10):
#     q = Queue()
#     p = Process(target=test, args=(q,))
#     p.start()
#     print p.pid, len(q.get())
# print('async')
# a = {}
# for i in range(10000):
#     a[str(i*1000000)] = [i]*20
# write_time = time()
# with open('test.pkl', 'wb') as f:
#     cPickle.dump(a, f, 2)
#
# print('write time', time()-write_time)
#
# load_time = time()
# with open('test.pkl', 'rb') as f:
#     b = cPickle.load(f)
# print(time() - load_time)

# mp_time = time()
# for i in range(4):
#     p = Process(target=copy_test, args=(b))
# print('mp time ', time() - mp_time)
#
# print(1/2, 1//2)
# print(np.random.random())

a = [0] * 9
a[0] = 1
a[1] = 2
a[2] = 3


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


# print(''.join(map(str, a)))
#
# prob = np.array([  3.51820181e-02,  6.10798925e-03,   4.88639140e-04,   3.90911312e-03,
#    9.54312240e-01])
# print(prob[0]>0.2)
# a = set()
# a.add(1)
# a.add(2)
# print(0 and 0 or 1 and 1)
# a = [[]]*3
# a[1] = [1, 2, 3]
# a[2] = [0, 1, 2]
# print(a)
#
# print(is_in_front([0, 0], 4, [0, 1]))
#
# a = np.random.random(5)
# print(np.std(a))


# np.random.seed(13)
# c = np.random.random(size=[3, 3, 2])
# d = np.random.random(size=[3, 3, 2])
# e = np.random.random(size=[3, 3, 2])
# f = copy(e)
# grid_input = np.array([[c, d], [e, f]])
# print(np.shape(grid_input))
# # grid_input = np.random.random(size=[2, 2, 3, 3, 2])
# with tf.Session() as sess:
#     tf.set_random_seed(13)
#     grid_placeholder = tf.placeholder(tf.float32, shape=[None, 2, 3, 3, 2])
#     conv1 = tf.nn.conv3d(input=grid_placeholder, filter=tf.get_variable(shape=[2, 2, 2, 2, 8], dtype=tf.float32, name='filter'), strides=[1, 1, 1, 1, 1], padding='VALID', name='conv1')
#     flat_conv1 = tf.contrib.layers.flatten(conv1)
#     sess.run(tf.global_variables_initializer())
#     v_a = tf.trainable_variables()
#     res = sess.run(conv1, feed_dict={grid_placeholder: grid_input})
#     print(np.shape(flat_conv1))
#     for w in v_a:
#         print(w)
# print([None] + list(np.shape([1, 2, 3])))
# x = np.random.random(size=[2, 100000000])
# # for i in range(10000000):
# #     x.append([2*i, 2*i+1])
# begin = time()
# np.random.shuffle(x)
# print(time() - begin)
# print(x)


# def one_hot(_index):
#     return np.eye(2)[np.reshape(_index, -1)][0]
#
#
# x = [[1, 2, 3, 4, 5], [2, 2, 3, 4, 5], [3, 2, 3, 4, 5], [4, 2, 3, 4, 5], [5, 2, 3, 4, 5]]
#
# N = 20
#
# dataset = tf.data.Dataset.from_tensor_slices({
#     'state': np.random.random(size=N),
#     'return': np.random.random(size=N),
#     'action': np.array(range(N))
# })
#
# dataset = dataset.repeat(2).shuffle(4).batch(2*N)
#
# iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()
# with tf.Session() as sess:
#     try:
#         while True:
#             print(sess.run(one_element))
#             # print('---------')
#             # sleep(0.1)
#     except tf.errors.OutOfRangeError:
#         print("end!")
# x = [1, 2, 3]
# x = np.reshape(x, [3, 1])
# print(x)
# r = np.repeat(x, 2, axis=1)
# print(r, r.shape)
# weights = np.load('weights_0.npy')
# for w in weights:
#     print(np.shape(w))

# begin = time()
# try:
#     r = xrange(100000)
# except NameError:
#     r = range(100000)
# print(sample(r, 10))
# print('time elapsed: ', time() - begin)
# state = [1, 2, 4, 2, 3, 4]
# id = 1
# print(state[id*3:id*3+3])
# print(cpu_count())
# a = np.array([1, 2, 3, 4], dtype=np.int64)
# np.save('haha.npy', a)
# c = np.load('haha.npy')
# print(c[c[2]])
#
# # ir_list = np.array(range(10, 110), dtype=np.int64)
# # for _ in range(5):
# #     np.save('{}.npy'.format(_), ir_list[20*_: 20*(_+1)])
# for _ in range(5):
#     print(np.load('{}.npy'.format(_)))


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


def update_policy(num_tfrecords=1, cur_iter=1):
    graph = tf.Graph()
    session = tf.Session(graph=graph, config=gpu_config)
    batch_size = _batch_size
    n_epochs = _num_epochs

    lr = 1e-2
    num_gpus = _num_gpu
    logdir = 'save/test/{}'.format(cur_iter)
    c1 = 0
    c2 = 1.0
    c3 = 0.0

    #os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
    no_gpu = not is_gpu_available()
    if no_gpu:
        num_gpus = 1
    with session as sess:
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        tower_grads = []
        tower_vars = []
        tf_record_list = ['episodes/0/{}.tfrecords'.format(i) for i in xrange(2000, 2000+num_tfrecords*cur_iter)]
        # print(tf_record_list)
        filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=n_epochs, shuffle=True)
        state, cu_re, label, act_prob = decode_from_tfrecords(filename_queue, batch_size=batch_size * num_gpus)
        act = tf.cast(label, tf.int32)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        global_iteration = tf.Variable(1, name='global_iteration', trainable=False)
        train_explosion = False
        # lr = tf.train.exponential_decay(lr, global_step, 500, 0.9985, staircase=True)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/cpu:0' if no_gpu else '/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                        h_state = tf.placeholder(tf.float32, [None, (_height + _width + 4) + (9 + _height + _width) * _num_players + 3], name='h_state')
                        h_act = tf.placeholder(tf.int32, [None, 1], name='target_action{}'.format(i))
                        # h_act_prob = tf.placeholder(tf.float32, [None, 5], name='target_action_prob{}'.format(i))
                        h_cu_re = tf.placeholder(tf.float32, [None, 1], name='cumulated_reward{}'.format(i))  # return from current state
                        h_cu_re_a = tf.placeholder(tf.float32, [None, 5], name='cumulated_reward{}'.format(i))  # return from current after taking some action
                        out, t_out, h_input, update_params, update_target_params, get_weights = build_training_model(h_state, 11, 5)
                        # qv_value = out[:, 5: 11]
                        cross_entropy_mean = tf.nn.softmax_cross_entropy_with_logits(logits=out[:, 0:5],
                                                                                     labels=tf.one_hot(h_act, 5))
                        pi_loss = tf.reduce_mean(cross_entropy_mean)
                        if tf.__version__ >= '2.3':  # just use mse
                            q_loss = tf.losses.huber_loss(predictions=out[:, 5:10], labels=h_cu_re_a, delta=1.0)
                            v_loss = tf.losses.huber_loss(predictions=out[:, 10:11], labels=h_cu_re, delta=1.0)
                        else:
                            q_loss = tf.losses.mean_squared_error(predictions=out[:, 5:10], labels=h_cu_re_a)
                            v_loss = tf.losses.mean_squared_error(predictions=out[:, 10:11], labels=h_cu_re)

                        # pi_logits = tf.nn.softmax(logits=self.out[:, 0:6])
                        # entropy = -tf.reduce_mean(tf.reduce_sum(-tf.log(pi_logits) * pi_logits, axis=1))  # negative of entropy term so that can be directly added in the loss function which is minimized
                        h_c2 = tf.placeholder(tf.float32, [1], name='coefficient_q')
                        h_c3 = tf.placeholder(tf.float32, [1], name='coefficient_pi')
                        loss = tf.reduce_mean(c1 * v_loss + h_c2 * q_loss + h_c3 * pi_loss, name='tower{}_loss'.format(i))
                        tower_vars.append(([h_state, h_cu_re, h_act, h_cu_re_a, h_c2, h_c3], q_loss,
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
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Group all updates to into a single train op.
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # to ensure testing phase works properly
        # with tf.control_dependencies(update_ops):
        train_op = apply_gradient_op

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)
        writer = tf.summary.FileWriter(logdir, graph)

    # with self.session as sess, sess.as_default():
    #     print(self.get_weights()[2][1][0:4])
    #     update_params(sess, get_weights())
    #     update_target_params(sess, self.get_weights())
        begin = time()
        step = 0
        shift_step = 0
        # if not os.path.exists('global_step'):
        #     shift_step = 0
        # else:
        #     with open('global_step', 'rb') as gs_f:
        #         shift_step = int(gs_f.read())
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
                # print(np.shape(_state))
                _feed_qv = []
                for _k in range(num_gpus):  # [h_state, h_cu_re, h_act, h_cu_re_a]
                    _t_f[tower_holders[_k][0]] = _state[_k * batch_size: (_k + 1) * batch_size]
                    # _feed_qv.append(tower_QVs[_k](_state[_k * batch_size: (_k + 1) * batch_size]))
                # _feed_qv = np.squeeze(_feed_qv)
                for _k in range(num_gpus):
                    _g = np.reshape(_cu_re[_k * batch_size: (_k + 1) * batch_size], [batch_size, 1])  # the return g_k = \sum \gamma^k r_{n+k}
                    # print(np.shape(_g))
                    _action_taken = _act[_k * batch_size: (_k + 1) * batch_size]
                    _t_f[tower_holders[_k][1]] = _g  # to fit ( T-1 iter + this iter )
                    _t_f[tower_holders[_k][2]] = _action_taken
                    _t_f[tower_holders[_k][3]] = _action_taken * _g.repeat(5, axis=1)  # np.eye(C)[index] return one_hot array of index
                    _t_f[tower_holders[_k][4]] = np.array([c2])
                    _t_f[tower_holders[_k][5]] = np.array([c3])
                # logger.info('Time for fetch and feed dict Step %d: %.5f sec a step' % (step+1, time.time()-start_time))
                if (step + 1) % 100 == 0:
                    # start_time = time.time()
                    _, t_q_loss, t_pi_loss, summary = sess.run([train_op, tower_q_losses, tower_pi_loss, summary_op], feed_dict=_t_f)
                    writer.add_summary(summary, shift_step+step)
                    # print(np.mean(t_pi_loss)/np.mean(t_q_loss))
                    # if np.mean(t_pi_loss)/np.mean(t_q_loss) > 1:
                    #     c2 = min([np.mean(t_pi_loss)/np.mean(t_q_loss)*c3, 5000.0])
                    # else:
                    #     c2 = max([np.mean(t_pi_loss)/np.mean(t_q_loss)*c3, 0.5])

                    duration = time() - start_time
                    if (step+1) % 2000 == 0:
                        logger.info('Step %d: %.5f sec a step' % (step + 1, duration))
                        # print(c2, c3)
                else:
                    _ = sess.run([train_op], feed_dict=_t_f)
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
            # print(t_weights[0][0])
            np.save('weights_.npy', get_weights(sess))
            with open('traintime.log', 'a+') as runlogfile:
                runlogfile.write('total time elapsed: %.2f min\n' % ((time() - begin) / 60.0))
            with open('global_step', 'wb+') as global_step_f:
                global_step_f.write(str(step+shift_step))
        finally:
            coord.request_stop()
            print('Training Done')
        coord.join(threads)
    return t_weights


def table_update():
    u_s = {}
    u_sa = {}
    average_strategy = {}

    graph = tf.Graph()
    session = tf.Session(graph=graph, config=gpu_config)
    num_records = 4000
    with session as sess:
        tf_record_list = ['episodes/0/{}.tfrecords'.format(i) for i in xrange(num_records)]
        # print(tf_record_list)
        filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=1, shuffle=False)
        state, cu_re, label, act_prob = decode_from_tfrecords(filename_queue, batch_size=1)
        act = tf.cast(label, tf.int32)

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        begin = time()
        try:
            while not coord.should_stop():
                _state, _cu_re, _act = sess.run([state, cu_re, act])  # 'state', 'return', 'action'
                _state = _state[0]
                # print(_state, _act[0], _cu_re)
                _state[-3:] *= _ammo
                _state[-3] = round(_state[-3])
                _state[-2] = round(_state[-2])
                _state[-1] = round(_state[-1])
                _item = [''.join(map(str, np.array(_state, dtype=np.int32))), _act[0][0], _cu_re[0][0]]
                # if _cu_re[0][0] > 0:
                #     print(_item)
                if _item[0] not in u_s:
                    u_s[_item[0]] = [_item[2], 1]
                else:
                    u_s[_item[0]][0] += _item[2]
                    u_s[_item[0]][1] += 1
                s_a_key = _item[0] + str(_item[1])
                if s_a_key not in u_sa:
                    u_sa[s_a_key] = [_item[2], 1]
                else:
                    u_sa[s_a_key][0] += _item[2]
                    u_sa[s_a_key][1] += 1
                if _item[0] not in average_strategy:
                    ac_v = [0] * 5
                    ac_v[_item[1]] = 1
                    average_strategy[_item[0]] = ac_v
                else:
                    average_strategy[_item[0]][_item[1]] += 1
        except tf.errors.OutOfRangeError:
            logger.info('total time elapsed: %.2f min' % ((time() - begin) / 60.0))
        finally:
            coord.request_stop()
            print('Update Done')
            with open('v{}.pkl'.format(num_records), 'wb') as f:
                cPickle.dump(u_s, f, 2)
            with open('q{}.pkl'.format(num_records), 'wb') as f:
                cPickle.dump(u_sa, f, 2)
            with open('pi{}.pkl'.format(num_records), 'wb') as f:
                cPickle.dump(average_strategy, f, 2)
        coord.join(threads)


def comparison():
    # u_s = {}
    # u_sa = {}
    # average_strategy = {}
    num_records = 4000
    with open('v{}.pkl'.format(num_records), 'rb') as f:
        u_s = cPickle.load(f)
    with open('q{}.pkl'.format(num_records), 'rb') as f:
        u_sa = cPickle.load(f)
    with open('pi{}.pkl'.format(num_records), 'rb') as f:
        average_strategy = cPickle.load(f)

    graph = tf.Graph()
    session = tf.Session(graph=graph, config=gpu_config)

    with session as sess:
        tf_record_list = ['episodes/0/{}.tfrecords'.format(i) for i in xrange(num_records)]
        # print(tf_record_list)
        filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=1, shuffle=False)
        state, cu_re, label, act_prob = decode_from_tfrecords(filename_queue, batch_size=1)
        act = tf.cast(label, tf.int32)

        h_state = tf.placeholder(tf.float32, [None, (_height + _width + 4) + (9 + _height + _width) * _num_players + 3], name='h_state')
        h_act = tf.placeholder(tf.int32, [None, 1], name='target_action{}'.format(i))
        # h_act_prob = tf.placeholder(tf.float32, [None, 5], name='target_action_prob{}'.format(i))
        h_cu_re = tf.placeholder(tf.float32, [None, 1], name='cumulated_reward{}'.format(i))  # return from current state
        h_cu_re_a = tf.placeholder(tf.float32, [None, 5], name='cumulated_reward{}'.format(i))  # return from current after taking some action
        out, t_out, h_input, update_params, update_target_params, get_weights = build_training_model(h_state, 11, 5)
        action_prob = tf.nn.softmax(out[:, 0:5])
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        begin = time()
        trained_weights = np.load('weights_.npy')
        update_params(sess, trained_weights)
        # print(trained_weights[0][0])
        ratio_q = [0, 0, 0, 0]
        ratio_v = [0, 0, 0, 0]
        kl = [0, 0]
        try:
            while not coord.should_stop():
                _state, _act, _cu_re = sess.run([state, act, cu_re])  # 'state', 'return', 'action'
                _out, _q = sess.run([out, action_prob], feed_dict={h_state: _state})
                raw_out = _out[0]
                q = _q[0]
                _state = _state[0]
                # print(_state, _act[0], _cu_re)
                _state[-3:] *= _ammo
                _state[-3] = round(_state[-3])
                _state[-2] = round(_state[-2])
                _state[-1] = round(_state[-1])
                _item = [''.join(map(str, np.array(_state, dtype=np.int32))), _act[0][0], 0]
                s_a_key = _item[0] + str(_item[1])
                if u_sa[s_a_key][0] != 0:
                    ratio_q[0] += math.fabs(1 - raw_out[5+_act[0][0]]/(u_sa[s_a_key][0]/u_sa[s_a_key][1] + 1e-2))
                    ratio_q[1] += 1
                else:
                    ratio_q[0] += math.fabs(raw_out[5 + _act[0][0]])
                    ratio_q[1] += 1

                if _cu_re[0][0] != 0:
                    ratio_q[2] += math.fabs(1 - raw_out[5+_act[0][0]]/(_cu_re[0][0]))
                    ratio_q[3] += 1
                # if random() < 1e-2:
                #     print(u_s[_item[0]][0], u_s[_item[0]][1])
                if u_s[_item[0]][0] != 0:
                    ratio_v[0] += math.fabs(1 - raw_out[-1]/(u_s[_item[0]][0] / u_s[_item[0]][1] + 1e-2))
                    ratio_v[1] += 1
                else:
                    ratio_v[0] += math.fabs(raw_out[-1])
                    ratio_v[1] += 1

                if _cu_re[0][0] != 0:
                    ratio_v[2] += math.fabs(1 - raw_out[-1]/(_cu_re[0][0] + 1e-2))
                    ratio_v[3] += 1
                # print(np.array(average_strategy[_item[0]], dtype=np.float64))
                p = np.array(average_strategy[_item[0]], dtype=np.float64)/sum(average_strategy[_item[0]])
                # print(p)
                # q = raw_out[0:5]
                # print(q)
                kl[0] += sum(p*np.log((p+1e-12)/(q+1e-12)))
                kl[1] += 1
                # if u_s[_item[0]][0] != 0:
                #     print(u_s[_item[0]][0] / u_s[_item[0]][1], raw_out[-1], u_sa[s_a_key][0]/u_sa[s_a_key][1], raw_out[5+_act[0][0]])
        except tf.errors.OutOfRangeError:
            logger.info('total time elapsed: %.2f min' % ((time() - begin) / 60.0))
        # finally:
        coord.request_stop()
        print('Update Done')
        print('ratio q: ', ratio_q[0]/ratio_q[1], ratio_q[2]/ratio_q[3], ratio_q[1], ratio_q[3], 'ratio v: ', ratio_v[0]/ratio_v[1], ratio_v[2]/ratio_v[3], ratio_v[1], ratio_v[3], 'kl: ', kl[0], kl[0]/kl[1])
        coord.join(threads)


def data_dis():
    num_records = 2000

    graph = tf.Graph()
    session = tf.Session(graph=graph, config=gpu_config)

    with session as sess:
        tf_record_list = ['episodes/0/{}.tfrecords'.format(i) for i in xrange(num_records)]
        # print(tf_record_list)
        filename_queue = tf.train.string_input_producer(tf_record_list, num_epochs=1, shuffle=False)
        state, cu_re, label, act_prob = decode_from_tfrecords(filename_queue, batch_size=1)
        act = tf.cast(label, tf.int32)

        h_state = tf.placeholder(tf.float32, [None, (_height + _width + 4) + (9 + _height + _width) * _num_players + 3],
                                 name='h_state')
        h_act = tf.placeholder(tf.int32, [None, 1], name='target_action{}'.format(i))
        # h_act_prob = tf.placeholder(tf.float32, [None, 5], name='target_action_prob{}'.format(i))
        h_cu_re = tf.placeholder(tf.float32, [None, 1],
                                 name='cumulated_reward{}'.format(i))  # return from current state
        h_cu_re_a = tf.placeholder(tf.float32, [None, 5],
                                   name='cumulated_reward{}'.format(i))  # return from current after taking some action
        out, t_out, h_input, update_params, update_target_params, get_weights = build_training_model(h_state, 11, 5)
        action_prob = tf.nn.softmax(out[:, 0:5])
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        begin = time()
        total = 0
        try:
            while not coord.should_stop():
                _state, _act, _cu_re = sess.run([state, act, cu_re])  # 'state', 'return', 'action'
                if _cu_re[0][0] != 0:
                    if np.abs(_cu_re[0][0]) == 1:
                        total += 1
        except tf.errors.OutOfRangeError:
            logger.info('total time elapsed: %.2f min' % ((time() - begin) / 60.0))
            print('non zero trajectory', total)
        # finally:
        coord.request_stop()
        coord.join(threads)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    return tf.train.Feature(int_list=tf.train.Int64List(value=value))


def simulation(players, q, _seed, cur_iter):
    import tensorflow as tf

    world = GridRoom(_seed)
    sampled_exp = []
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
        for _pid in range(_num_players):  # TODO : replace with player 1
            v = 0
            for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                players[_pid].exp_buffer[_k][2] += _gamma * v
                v = players[_pid].exp_buffer[_k][2]
            # if v < 0:
            #     print(players[_pid].exp_buffer)
            # print(players[_pid].exp_buffer)
            sampled_exp[_pid] += players[_pid].exp_buffer  # all agents share it if sp is used
            if _pid == 0:
                # print('write player')
                train_writer = tf.python_io.TFRecordWriter('episodes/{}/{}.tfrecords'.format(_pid, cur_iter*2000 + _i))
                v = 0
                for _k in range(len(players[_pid].exp_buffer) - 1, -1, -1):
                    _item = players[_pid].exp_buffer[_k]
                    # players[_pid].exp_buffer[_k][2] += _gamma * v
                    v = players[_pid].exp_buffer[_k][2]
                    one_hot_s = players[_pid].parse_state([_item[0][1], _item[0][3]])
                    full_state = one_hot_s + [_item[0][2] / float(_ammo)] * 3
                    b_feature = {'State': _float_feature(full_state),
                                 'Return': _float_feature([v]),
                                 'Act': _float_feature([_item[1]]), 'Act_prob': _float_feature([1.0])}
                    example = tf.train.Example(features=tf.train.Features(feature=b_feature))
                    train_writer.write(example.SerializeToString())
                train_writer.close()
    q.put(sampled_exp)


def update_pi(player, exp, q):
    player.update_policy(exp)
    q.put([player.u_s, player.u_sa, player.average_strategy])


def train():
    # world = GridRoom()
    players = [RMAgent(i) for i in range(_num_players)]
    # for _i in range(0):
    #     with open('v{}_3.999.pkl'.format(_i), 'rb') as f:
    #         players[_i].u_s = cPickle.load(f)
    #     with open('q{}_3.999.pkl'.format(_i), 'rb') as f:
    #         players[_i].u_sa = cPickle.load(f)
    #     with open('pi{}_3.999.pkl'.format(_i), 'rb') as f:
    #         players[_i].average_strategy = cPickle.load(f)
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
            p = Process(target=simulation, args=(copy(players), _q, None, _iteration+1))
            p.daemon = True
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
        # print(len(_exp[0][0]), len(_exp[1][0]))
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
        gc.collect()
        if (_iteration + 1) % 100 == 0:
            print('This is %d step, %.3f' % (_iteration, _iteration/_train_iter))
        if (_iteration + 1) % _save_fre == 0:
            for _pid in range(_num_players-1):
                with open('v{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[_pid].u_s, f, 2)
                with open('q{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[_pid].u_sa, f, 2)
                with open('pi{}_{}.pkl'.format(_pid, (_iteration+1)//_save_fre), 'wb') as f:
                    cPickle.dump(players[_pid].average_strategy, f, 2)
    print('Time eplapsed: %.2f' % (time() - begin))


def test(_iter):
    world = GridRoom()
    players = [ShootingAgent1(i) for i in range(_num_players)]
    # players[0] = ShootingAgent1(0)
    players[0] = NRMAgent(0)

    # players[0] = RMAgent(0)
    # # num_records = 4000
    # with open('v0_{}.0.pkl'.format(_iter), 'rb') as f:
    #     players[0].u_s = cPickle.load(f)
    # with open('q0_{}.0.pkl'.format(_iter), 'rb') as f:
    #     players[0].u_sa = cPickle.load(f)
    # with open('pi0_{}.0.pkl'.format(_iter), 'rb') as f:
    #     players[0].average_strategy = cPickle.load(f)

    # players.append(RandomAgent(1))
    # players[0] = ShootingAgent(0)
    # players[1] = ShootingAgent1(1)
    for _k in range(1, 2):
        players[0].seen = 0
        players[0].unseen = 0
        players[1].seen = 0
        players[1].unseen = 0
        for _i in range(_num_players-1):
            # begin = time()
            # # players[_i].update_weights(np.load('weights_.npy'))
            players[_i].update_weights(np.load('model/test/weights_{}.npy'.format(_iter)))
            players[_i].test = False
            players[_i].exploration = False
            # print('Time for load model: ', time() - begin, players[_i].u_s.__len__(), players[_i].u_sa.__len__(), players[_i].average_strategy.__len__())
        # players[1].test = False
        # players[1].exploration = True
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
        print('Time eplapsed: %.2f min' % ((time() - begin)/60.0), step/_test_iter)
        print(total_r)


# a = [1.0, 7.0, round(0.999999),1,2,3,4,5]
# print(a[-3:])
# a = np.array(a, dtype=np.int64)
# print(a)
# c = ''.join(map(str, a))
# print(c)
gc.collect()
# table_update()
# update_policy(num_tfrecords=4000)
# comparison()
# data_dis()
# test()
# a = np.array([0.1, 0.3, 0.6])
# c = np.array([0.2, 0.8-1e-12, 1e-12])
# print(c+1e-6, sum(-c*np.log(c)), sum(-c*np.log(c+1e-6)))
# print(a*np.log((a+1e-6)/(c+1e-6)), sum(a*np.log(a/c)), sum(a*np.log(((a+1e-12)/(1+3e-12))/((c+1e-12)/(1+3e-12)))))

np.array([0, 3, 5, 1, 3], dtype=np.float64)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-si', '--sample_iter', dest='sample_iter', default=100, type=int)
parser.add_argument('-t', '--thread', dest='thread', default=1, type=int, help='Number of thread to simulation')
args = parser.parse_args()
# print(args.thread, args.sample_iter)
if not os.path.exists('episodes'):
    os.mkdir('episodes')
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('model/test'):
    os.mkdir('model/test')
if not os.path.exists('episodes/0'):
    os.mkdir('episodes/0')

# train()

# for _iter in range(10, _train_iter - 39):
#     t_weights = update_policy(10000, _iter)
#     np.save('model/test/weights_{}.npy'.format(_iter), t_weights)
#
# # for _iter in range(1, 1 + _train_iter-40):
# #     test(_iter)
# #     print('----------------------------')
#
# for _iter in range(10, 11):
#     with open('v0_{}.0.pkl'.format(_iter), 'rb') as f:
#         u_s = cPickle.load(f)
#     with open('q0_{}.0.pkl'.format(_iter), 'rb') as f:
#         u_sa = cPickle.load(f)
#     with open('pi0_{}.0.pkl'.format(_iter), 'rb') as f:
#         average_strategy = cPickle.load(f)
#     agent = NRMAgent(0)
#     agent.update_weights(np.load('model/test/weights_{}.npy'.format(_iter)))
#     for _key in u_s.keys():
#         valid_action = 4
#         tmp_q = []
#         if int(_key[7]) > 0:
#             valid_action = 5
#         for i in range(5-valid_action, 5):
#             tmp_key = copy(_key)+str(i)
#             if tmp_key in u_sa.keys():
#                 sa = u_sa[tmp_key]
#             else:
#                 sa = [0, 1.0]
#             tmp_q.append(np.true_divide(sa[0], sa[1]))
#         # print(_key)
#         # print([list(map(int, list(_key[1:7]))), list(map(int, list(_key[7:9])))])
#         one_hot_s = agent.parse_state([list(map(int, list(_key[1:7]))), list(map(int, list(_key[7:9])))])
#         # print(one_hot_s + [int(_key[9])/float(_ammo)]*3)
#         tmp_s = one_hot_s + [int(_key[9]) / float(_ammo)] * 3
#         tmp_s = np.reshape(tmp_s, [-1, 37])
#         print(agent.get_cu_re(tmp_s)[0][0:5], tmp_q)


# a = '010203011'
# print(list(map(int, list(a[0:6]))))
c = np.random.random(size=[10, 32, 32])
print(c)