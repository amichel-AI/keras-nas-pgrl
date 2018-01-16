import numpy as np
from keras.layers import Activation
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout
from keras.layers.pooling import MaxPool2D, AvgPool2D
from keras.layers.normalization import BatchNormalization

class ActionHelper():

    def __init__(self, action_set='short'):
        self.action_set = action_set

        self.actions_name = []
        self.actions_info = []

        self.actions_type = []
        self.actions_params = []
        self._curr_action_num = 0
        self._this_action = None
        self._this_name = "None"
        self._this_name = ""
        self.initActionsList()

    @property
    def all_actions_length(self):
        return len(self.actions_name)

    def isNone(self, action):
        return (action == 0)

    def decode_layer_actionID(self, action):
        return action % self.all_actions_length, int(action / self.all_actions_length)

    def initActionsList(self):
        self.actions_name = ["None"]
        self.actions_info = [""]

        # Add all the other actions
        if (self.action_set == 'short'):
            return self.shortActionsList(None, None)

    def getAction(self, input, which_action):
        self._curr_action_num = 0

        # Add all the other actions
        if (self.action_set == 'short'):
            return self.shortActionsList(input, which_action)

    def shortActionsList(self, input, which_action):
        self._this_action = None
        self._this_info = ""
        self._conv2D_blocks(input, which_action, [8, 16, 32, 64], [(3, 3), (5, 5)], ['relu'])
        self._dropOut_blocks(input, which_action, [0.35, 0.5, 0.85, 0.95, 0.99])
        self._dense_blocks(input, which_action, [16, 32, 64, 128], ['relu'])
        self._normalization_blocks(input, which_action, )
        self._activation_blocks(input, which_action, ['relu', 'tanh', 'selu', 'sigmoid'])
        self._avgPooling_blocks(input, which_action, [(3, 3), (5, 5)])
        self._maxPooling_blocks(input, which_action, [(3, 3), (5, 5)])
        return self._this_action, self._this_name, self._this_info

    #def explore(self):
    #    action_type = int(np.random.randint(0, len(self.actions_type)))
    #    which_params = int(np.random.randint(0, len(self.actions_params[action_type])))
    #    #print('JEJE: {} and {}'.format(action_type, which_params))
    #    return self.actions_params[action_type][which_params]

    def _conv2D_blocks(self, input, which_action, nb_channel_list, kernsize_list, activation_list, padding='same', init='glorot_normal'):
        if (input == None):
            self.actions_type.append('Conv2D')
            self.actions_params.append([])

        for i in nb_channel_list:
            for k in kernsize_list:
                for activation in activation_list:
                    self._curr_action_num += 1
                    if (input == None):
                        self.actions_name.append('Conv2D'.format(self._curr_action_num))
                        self.actions_info.append('{}@{}-{}'.format(i,k,activation))
                        self.actions_params[-1].append(self._curr_action_num)
                    else:
                        if (self._curr_action_num == which_action):
                            self._this_action = Conv2D(i, k, activation=activation, padding=padding, kernel_initializer=init)(input)
                            self._this_name = self.actions_name[self._curr_action_num]
                            self._this_info = self.actions_info[self._curr_action_num]
                            return self._this_action, self._this_name, self._this_info

    def _maxPooling_blocks(self, input, which_action, poolsize_list):
        if (input == None):
            self.actions_type.append('MaxPool')
            self.actions_params.append([])

        for p in poolsize_list:
            self._curr_action_num += 1
            if (input == None):
                self.actions_name.append('MaxPool'.format(self._curr_action_num))
                self.actions_info.append('{}'.format(p))
                self.actions_params[-1].append(self._curr_action_num)
            else:
                if (self._curr_action_num == which_action):
                    self._this_action = MaxPool2D(p)(input)
                    self._this_name = self.actions_name[self._curr_action_num]
                    self._this_info = self.actions_info[self._curr_action_num]
                    return self._this_action, self._this_name, self._this_info

    def _avgPooling_blocks(self, input, which_action, poolsize_list):
        if (input == None):
            self.actions_type.append('AvgPool')
            self.actions_params.append([])
        for p in poolsize_list:
            self._curr_action_num += 1
            if (input == None):
                self.actions_name.append('AvgPool'.format(self._curr_action_num))
                self.actions_info.append('{}'.format(p))
                self.actions_params[-1].append(self._curr_action_num)
            else:
                if (self._curr_action_num == which_action):
                    self._this_action = AvgPool2D(p)(input)
                    self._this_name = self.actions_name[self._curr_action_num]
                    self._this_info = self.actions_info[self._curr_action_num]
                    return self._this_action, self._this_name, self._this_info

    def _dropOut_blocks(self, input, which_action, dro_list):
        if (input == None):
            self.actions_type.append('DropOut')
            self.actions_params.append([])

        for d in dro_list:
            self._curr_action_num += 1
            if (input == None):
                self.actions_name.append('DropOut'.format(self._curr_action_num))
                self.actions_info.append('{:2}'.format(d))
                self.actions_params[-1].append(self._curr_action_num)
            else:
                if (self._curr_action_num == which_action):
                    self._this_action = Dropout(d)(input)
                    self._this_name = self.actions_name[self._curr_action_num]
                    self._this_info = self.actions_info[self._curr_action_num]
                    return self._this_action, self._this_name, self._this_info

    def _dense_blocks(self, input, which_action, outsize_list, activation_list, init='he_uniform'):
        if (input == None):
            self.actions_type.append('Dense')
            self.actions_params.append([])

        for o in outsize_list:
            for activation in activation_list:
                self._curr_action_num += 1
                if (input == None):
                    self.actions_name.append('Dense'.format(self._curr_action_num))
                    self.actions_info.append('{}-{}'.format(o, activation))
                    self.actions_params[-1].append(self._curr_action_num)
                else:
                    if (self._curr_action_num == which_action):
                        self._this_action = Dense(o, kernel_initializer=init)(input)
                        self._this_name = self.actions_name[self._curr_action_num]
                        self._this_info = self.actions_info[self._curr_action_num]
                        return self._this_action, self._this_name, self._this_info

    def _normalization_blocks(self, input, which_action):
        if (input == None):
            self.actions_type.append('BN')
            self.actions_params.append([])

        self._curr_action_num += 1
        if (input == None):
            self.actions_name.append('BN'.format(self._curr_action_num))
            self.actions_info.append("")
            self.actions_params[-1].append(self._curr_action_num)
        else:
            if (self._curr_action_num == which_action):
                self._this_action = BatchNormalization()(input)
                self._this_name = self.actions_name[self._curr_action_num]
                self._this_info = self.actions_info[self._curr_action_num]
                return self._this_action, self._this_name, self._this_info

    def _activation_blocks(self, input, which_action, activations_list):
        if (input == None):
            self.actions_type.append('Act')
            self.actions_params.append([])

        for a in activations_list:
            self._curr_action_num += 1
            if (input == None):
                self.actions_name.append('Act'.format(self._curr_action_num))
                self.actions_info.append('{}'.format(a))
                self.actions_params[-1].append(self._curr_action_num)
            else:
                if (self._curr_action_num == which_action):
                    self._this_action = Activation(a)(input)
                    self._this_name = self.actions_name[self._curr_action_num]
                    self._this_info = self.actions_info[self._curr_action_num]
                    return self._this_action, self._this_name, self._this_info