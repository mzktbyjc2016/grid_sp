#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 5/30/18 6:35 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''

import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np


class Model(object):

    def __init__(self, graph):
        self.graph = graph

    def create_model(self, session, input_dim, out_dim, action_dim):
        with self.graph.as_default():
            if type(input_dim) is int:
                input_dim = [input_dim]
            h_input = tf.placeholder(tf.float32, shape=[None] + input_dim, name='h_input')
            with tf.variable_scope('pi'):
                dense1 = tf.layers.dense(h_input, 128,
                                         activation=tf.nn.leaky_relu,
                                         use_bias=True,
                                         kernel_initializer=None,
                                         bias_initializer=None,
                                         kernel_regularizer=None,
                                         bias_regularizer=None,
                                         activity_regularizer=None,
                                         trainable=True,
                                         name='dense1', )
                dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.leaky_relu, use_bias=True, bias_initializer=init_ops.zeros_initializer, name='dense2')

                out = tf.layers.dense(dense2, out_dim, use_bias=True, activation=None, name='out')

            action_prob = tf.nn.softmax(out[:, 0: action_dim])
            qv_value = out[:, action_dim: 2*action_dim+1]
            q_value = qv_value[:, 0: action_dim]
            v_value = qv_value[:, action_dim: action_dim+1]
            regret_plus = tf.nn.relu(q_value - v_value)

            with tf.variable_scope('target'):
                t_dense1 = tf.layers.dense(h_input, 128,
                                         activation=tf.nn.leaky_relu,
                                         use_bias=True,
                                         kernel_initializer=None,
                                         bias_initializer=None,
                                         kernel_regularizer=None,
                                         bias_regularizer=None,
                                         activity_regularizer=None,
                                         trainable=True,
                                         name='dense1', )
                t_dense2 = tf.layers.dense(t_dense1, 128, activation=tf.nn.leaky_relu, use_bias=True, bias_initializer=init_ops.zeros_initializer, name='dense2')

                t_out = tf.layers.dense(t_dense2, out_dim, use_bias=True, activation=None, name='out')
            pi_weights_v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pi')
            pi_update_placeholder = []
            pi_update_op = []
            for i, _ in enumerate(pi_weights_v):
                pi_update_placeholder.append(tf.placeholder(_.dtype, shape=_.get_shape()))
                pi_update_op.append(_.assign(pi_update_placeholder[i]))
            a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
            update_placeholder = []
            update_op = []
            for i, _ in enumerate(a):
                update_placeholder.append(tf.placeholder(_.dtype, shape=_.get_shape()))
                update_op.append(_.assign(update_placeholder[i]))
            session.run(tf.global_variables_initializer())

        def update_params(weights):
            try:
                for k, __ in enumerate(pi_update_op):
                    session.run(__, feed_dict={pi_update_placeholder[k]: weights[k]})
            except IndexError:
                print(len(weights), len(pi_update_op))

        def get_weights():
            return session.run(pi_weights_v)

        def update_target_params(weights):
            try:
                for k, __ in enumerate(update_op):
                    session.run(__, feed_dict={update_placeholder[k]: weights[k]})
            except IndexError:
                print(len(weights), len(update_op))

        def get_average_policy(state):
            return session.run(action_prob, feed_dict={h_input: state})[0]

        def get_behaviorial_policy(state):
            tmp_state = np.reshape(state, [1] + input_dim)
            regret_p = session.run(regret_plus, feed_dict={h_input: tmp_state})[0]
            # print(regret_p)
            if sum(regret_p) > 0:
                return np.true_divide(regret_p, np.sum(regret_p))
            else:
                return np.true_divide([1.0] * action_dim, action_dim)

        def get_cumulative_return(state):
            """
            This is used to get the T-1 iteration expected cumulative return
            :param state: a extended state with raw pixel (already in range [0, 1]) and extra game info
            :return: T-1 expected cumulative return
            """
            total_cu_re = session.run(qv_value, feed_dict={h_input: state})
            return total_cu_re

        return out, t_out, h_input, get_weights, update_params, get_average_policy, get_behaviorial_policy, update_target_params, get_cumulative_return
    #
    # def create_target_model(self, input_dim, out_dim, action_dim):
    #     # with self.graph.as_default():
    #     if 1:
    #         if type(input_dim) is int:
    #             input_dim = [input_dim]
    #         h_input = tf.placeholder(tf.float32, shape=[None] + input_dim)
    #         with tf.variable_scope('target'):
    #             dense1 = tf.layers.dense(h_input, 128,
    #                                      activation=tf.nn.leaky_relu,
    #                                      use_bias=True,
    #                                      kernel_initializer=None,
    #                                      bias_initializer=None,
    #                                      kernel_regularizer=None,
    #                                      bias_regularizer=None,
    #                                      activity_regularizer=None,
    #                                      trainable=True,
    #                                      name='dense1', )
    #             dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.leaky_relu, use_bias=True, bias_initializer=init_ops.zeros_initializer, name='dense2')
    #
    #             out = tf.layers.dense(dense2, out_dim, use_bias=True, activation=None, name='out')
    #         a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
    #         update_placeholder = []
    #         update_op = []
    #         for i, _ in enumerate(a):
    #             update_placeholder.append(tf.placeholder(_.dtype, shape=_.get_shape()))
    #             update_op.append(_.assign(update_placeholder[i]))
    #
    #     def update_target_params(session, weights):
    #         for k, __ in enumerate(update_op):
    #             session.run(__, feed_dict={update_placeholder[k]: weights[k]})
    #
    #     return out, update_target_params