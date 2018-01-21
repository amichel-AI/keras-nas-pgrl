import os
import sys
import time
import math
import numpy as np
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from net_reflection import NetReflection
from actions import ActionHelper
from caching import CachingHelper
import random
from collections import deque

        #create_policy_network(self):
        # define the network mu (input=s, output=s)

        # logprobs = logprobs prediction from mu(s)
        # pg_loss = reduce_mean_cross_entropy_loss (logprobs)
        # reg_term = regularize the variables in the network mu(s)

        # Define loss here
        #L = reg_param*reg_term + pg_loss

        # Compute reward
        # V(s,mu) <=> testing loss = reflect the NAS from s and train it to get the testing loss

        # PG is
        # compute_gradients(L) <=> grad(L) = grad( r*rt + logprobs(mu(s)) )
        # include rewards <=> E[V^mu(s) x grad(L)]

class PGAgent:
    def __init__(self, environment):
        self.env = environment
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.33  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()
        #self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_size))
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='sigmoid'))

        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward, next_state):
        y = np.zeros([self.action_size])
        y[action] = 1
        grad = (np.array(y).astype('float32') - prob)
        this_state = np.copy(state)
        this_state[:] = state[:]
        self.memory.append((this_state, action, prob, reward, next_state, grad))

    def policy(self, state, prev_action):
        state = state.reshape([1, 1, state.shape[-1]])
        all_probs = self.model.predict(state, batch_size=1).flatten()
        probs   = all_probs / np.sum(all_probs)

        if np.random.rand() >= self.epsilon:
            action = np.random.choice(self.action_size, 1, p=probs)[0]
            if action == prev_action:
                action = self.explore()
        else:
            action = self.explore()

        return action, probs

    def explore(self):
        return self.env.explore()

    def _sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    def train(self, batch_size=100):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        X, Y = [], []
        i = 0
        for state, action, prob, reward, next_state, gradient in minibatch:
            gradient *= reward

            #state = state.reshape([1, state.shape[-1]])
            my_state = state.reshape([1, state.shape[-1]])
            X.append(my_state)

            Y_corr = prob + self.learning_rate*gradient # is it SGD
            Y_corr = self._sigmoid(Y_corr)
            Y_corr = Y_corr.reshape([1,self.env.action_size])
            Y.append(Y_corr)

            #print('State: {} | Next: {}'.format(state, next_state))
            #X=np.array(np.stack(X))
            #self.model.train_on_batch(X, Y)
            #X, Y = [], []

        X = np.array(np.stack(X))
        ##X = np.reshape(X, [None, 1,  X.shape[2]])
        ##Y = np.stack(Y)
        ##Y = np.reshape(Y, [1, Y.shape[0], Y.shape[2]])
        self.model.train_on_batch(X, Y)
        self.memory.clear()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, max_episodes=10, batch_size=100):
        level = 1
        last_action = 0
        this_state = np.copy(self.env.current_state)

        ee=0
        for e in range(max_episodes):
            this_state[:] = self.env.current_state[:]

            good_probe = False
            repeat = False
            while (1): # do an action until the state changes
                # get next state according to current policy function
                action, prob = self.policy(this_state, last_action)

                # perform the selected action/prob
                t = time.time()
                reward, next_state, loss, acc, test_loss, test_acc, repeated = self.env.step(action, prob, level)
                elapsed = time.time() - t
                if (reward > 0) and (not repeated): # if so then this was a successfull probe so count one episode more
                    good_probe = True
                    ee += 1
                break
            print('[{:.2f} secs] Episode {}[{}](level={}) complete after action {} training with {:.2f} epochs => reward: {:.2f}, Train loss:{:.4f}, Train acc: {:.2f}, Test acc: {:.4f}'.format(elapsed,e,ee,level,action,self.env.nasconfig.max_epochs,reward,loss,acc,test_acc))
            sys.stdout.flush()

            # keep results for further training
            self.remember(this_state, action, prob, reward, next_state)

            # update the policy according to the reward
            self.train(batch_size=batch_size)
            last_action = action

            # next level ?
            if good_probe and (ee % self.env.nasconfig.level_episode_step == 0):
                self.env.nasconfig.max_epochs *= self.env.nasconfig.epochs_increase
                if (self.env.nasconfig.reset_state):
                    state = self.env.best_state
                level += 1

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class NASenv():
    def __init__(self, nasconfig, trainData, testData, nb_classes, train_callback, test_callback):
        self.nasconfig = nasconfig
        self.env_actions = ActionHelper()
        self.caching = CachingHelper(nasconfig, cache=True)
        self.nb_classes = nb_classes
        self.trainData = trainData
        self.testData = testData
        self.train_callback = train_callback
        self.test_callback = test_callback
        self.state = np.zeros(self.state_size[-1])
        #self.state[0], self.state[-1] = 1, self.action_size

    @property
    def current_state(self):
        return self.state

    @property
    def all_actions_length(self):
        return self.env_actions.all_actions_length

    @property
    def state_size(self):
        return [1,self.nasconfig.net_width*self.nasconfig.net_layers]

    @property
    def action_size(self):
        return self.state_size[-1]*self.all_actions_length

    @property
    def best_state(self):
        return self.caching.getBestState()

    def observe_architecture(self, state):
        this_arch = np.zeros(shape=[self.nasconfig.net_layers, self.nasconfig.net_width], dtype=int)
        for i in range(len(state)):
            action, pos_in_grid = state[i], i
            layer, branch = pos_in_grid % self.nasconfig.net_layers, int(pos_in_grid / self.nasconfig.net_layers)
            this_arch[layer, branch] = int(action)
        return self.state_from_arch(state, this_arch)

    def state_from_arch(self, state, arch):
        new_state = np.zeros(state.shape)
        new_arch = np.zeros(arch.shape, dtype=int)
        nb_b = np.zeros(self.nasconfig.net_width, dtype=int)

        for j in range(self.nasconfig.net_width):
            for i in range(self.nasconfig.net_layers):
                action = arch[i, j]
                if (not self.env_actions.isNone(action)):
                    new_arch[int(nb_b[j]), j] = action
                    p = j*self.nasconfig.net_layers + nb_b[j]
                    new_state[p] = action
                    nb_b[j] += 1

        return new_state, new_arch

    def explore(self):
        #which_action = self.env_actions.explore()
        #which_layer = int(np.random.randint(0, self.state_size[-1]))
        ##print('Hence {} in layer {}'.format(self.env_actions.actions_name[which_action], which_layer))
        #return which_action+which_layer*self.all_actions_length
        return int(np.random.randint(0, self.action_size))

    def step(self, action, prob, level=0):
        actionID, node_in_graph = self.env_actions.decode_layer_actionID(action)

        trans_state = np.copy(self.current_state)
        trans_state[:] = self.current_state[:]
        trans_state[node_in_graph] = int(actionID)

        # train the new architecture defined in "next_state"
        (x_train, y_train) = self.trainData
        (x_test, y_test) = self.testData

        next_state, this_arch = self.observe_architecture(trans_state)
        if not (next_state == self.current_state).all():
            nasmodel = NetReflection(len(x_train), x_train.shape[1:], self.nb_classes, this_arch, self.env_actions, action, self.nasconfig)
            self.current_state[:] = next_state[:]
            repeat = False
        else:
            repeat = True

        if (not repeat) and (nasmodel.model != None):
            # Caching sols
            incache, loss, acc, test_loss, test_acc = self.caching.fromcache(self.current_state, level)
            if not incache:
                loss, acc = self.train_callback(nasmodel.model, x_train, y_train, x_test, y_test)
                test_loss, test_acc = self.test_callback(nasmodel.model, x_test, y_test)
                self.caching.tocache(self.current_state, level, loss, acc, test_loss, test_acc)

            reward = acc
        else:
            acc = 0
            loss = 100
            test_acc = 0
            test_loss = 100
            reward = -1

        return reward, next_state, loss, acc, test_loss, test_acc, repeat
