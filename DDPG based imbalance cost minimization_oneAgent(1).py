"""
In this version, the data used are 2006-2010.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



import numpy as np
import sys
import random

import matplotlib.pyplot as plt

from Environment import Environment
import tensorflow as tf




#####################  import the content in console to files  ####################
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger("DDPG-Result.txt") # aim to minimize the local voltage deviation.sub-1. lc=0.05. Minimize local voltage deviation. maximum reactive power is variable.

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], name="s")
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.index = 0

        self.a = self._build_a(self.S,)
        self.a_mid = tf.multiply(self.a, 1, name="get_action")
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)
        self.test = a_loss

        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)
            # self.td_error = td_error

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        self.index += 1
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        if self.index % 1000 == 0:
            self.saver.save(self.sess, './sub-model-1', global_step=self.index + 1)  # 新加

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None, ):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net_layer1 = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_layer2 = tf.layers.dense(net_layer1, 100, activation=tf.nn.relu, name='l2', trainable=trainable)
            # net_layer3 = tf.layers.dense(net_layer2, 32, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net_layer2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name="scaled_a")

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_layer1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            n_l2 = 100
            w2_a = tf.get_variable('w2_a', [n_l1, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net_layer2 = tf.nn.relu(tf.matmul(net_layer1, w2_a) + b2)
            # n_l3 = 32
            # w3_a = tf.get_variable('w3_a', [n_l2, n_l3], trainable=trainable)
            # b3 = tf.get_variable('b3', [1, n_l3], trainable=trainable)
            # net_layer3 = tf.nn.relu(tf.matmul(net_layer2, w3_a) + b3)
            return tf.layers.dense(net_layer2, 1, trainable=trainable)  # Q(s,a)


#####################  hyper parameters  ####################

MAX_EPISODES = 12000
MAX_EP_STEPS = 10
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0    # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

s_dim = 3
a_dim = 3
a_bound = 1


env = Environment()
ddpg = DDPG(a_dim, s_dim, a_bound)




x = []
y = []
plt.figure(figsize=(10, 8))

plt.ion()


###############################  training  ####################################
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    # action_set_1 = np.zeros((24, 4))
    # action_set_2 = np.zeros((24, 4))

    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        r = np.zeros(1)

        if ddpg.pointer < MEMORY_CAPACITY:
            a = [random.uniform(-1, 1) for m in range(a_dim)]
            a = np.asarray(a)
        else:
            a = ddpg.choose_action(s)  # add randomness to action selection for exploration

        r = env.get_reward(a)
        s_ = env.get_state()

        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward_with_penalty: ', ep_reward)
            break





    x.append(i)
    y.append(ep_reward)
    plt.clf()
    plt.plot(x, y, color='b', linewidth=2)
    plt.pause(0.01)
    plt.ioff()


plt.show()
