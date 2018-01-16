import numpy as np
import keras as keras
from keras import backend as K
from keras import Sequential
from keras import Model
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.engine import InputLayer
from keras.layers import Activation, Dense, Flatten, Concatenate, Dropout, Merge, merge
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, AvgPool1D, AvgPool2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from actions import ActionHelper
import gc

class NetReflection():
    model = Sequential()

    def __init__(self, input_shape, num_output, architecture, env_actions, config):
        self.model = None
        gc.collect()
        self.combine_branches = config.net_merging

        # Preparing first-level nodes after root
        self.model = None

        net_name = "NAS_"
        input_layer = keras.layers.Input(shape=input_shape, name='input_4allbranches')

        branches = []
        for branch in range(config.net_width):
            branches.append(input_layer)
        is_branch_active = np.zeros([config.net_width])

        for layer in range(config.net_layers):
            for branch in range(config.net_width):
                if self.combine_branches:
                    if (is_branch_active[branch]) and (not env_actions.isNone(architecture[layer,branch])):
                        # combine adjacent branches
                        if branch > 0:
                            if branch < config.net_width - 1:
                                if not self.canMerge([branches[branch-1], branches[branch], branches[branch+1]]): return # Stop if layers can't be merged
                                branches[branch] =  Concatenate()([branches[branch - 1], branches[branch], branches[branch + 1]])
                            else:
                                if not self.canMerge([branches[branch-1], branches[branch]]): return # Stop if layers can't be merged
                                branches[branch] = Concatenate()([branches[branch - 1], branches[branch]])
                        else:
                            if branch < config.net_width - 1:
                                if not self.canMerge([branches[branch], branches[branch+1]]): return # Stop if layers can't be merged
                                branches[branch] = Concatenate()([branches[branch], branches[branch + 1]])

                if not env_actions.isNone(architecture[layer, branch]):
                    # Add the layer corresponding to the state
                    is_branch_active[branch] += 1
                    branches[branch], this_name, this_info = env_actions.getAction(branches[branch], architecture[layer,branch])
                    net_name += "_"+this_name+"["+this_info+"]"

        # finally combine all branches here
        if (config.net_width == 1):
            all_branches = branches[0]
            all_branches = Flatten()(all_branches)
        else:
            if not self.canMerge(branches): return # Stop if layers can't be merged
            for branch in range(config.net_width):
                branches[branch] = Flatten()(branches[branch])
            all_branches = Concatenate()(branches)

        # Last layer should be dense(classes)->softmax for classification
        all_branches = Dropout(0.85)(all_branches)
        all_branches = Dense(num_output)(all_branches)
        all_branches = Activation('softmax')(all_branches)
        self.model = Model(name=net_name, inputs=[input_layer], outputs=[all_branches])

        # Model optimization
        if (config.steps_per_action > 0):
            earlyStopping = EarlyStoppingBySGDSteps(max_sgd_steps=config.steps_per_action,
                                                batch_size=config.batch_size)
            self.model.callbacks=[earlyStopping]
        else:
            self.model.callbacks = []
        opt = Adam(lr=config.learning_rate)
        self.model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)

        # Model summary
        self.nb_trainable_params = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        self.nb_non_trainable_params = int(np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))
        print('Parameters => trainable:{:,}, total:{:,}'.format(self.nb_trainable_params, self.nb_trainable_params+self.nb_non_trainable_params)+" | Model [" + net_name+"]")
        #self.model.summary()

    def canMerge(self, layers):
        for i in range(len(layers)-1):
            shapeA = np.array(layers[i].get_shape().as_list())
            shapeB = np.array(layers[i+1].get_shape().as_list())
            #print('{},{}'.format(shapeA, shapeB))
            if (shapeA != shapeB).sum() > 1:
                return False
        return True

class EarlyStoppingBySGDSteps(Callback):
    def __init__(self, max_sgd_steps, batch_size):
        super(Callback, self).__init__()
        self.max_sgd_steps = max_sgd_steps
        self.batch_size = batch_size

    def on_batch_end(self, batch, logs={}):
        if self.max_sgd_steps <= batch*self.batch_size:
            self.model.stop_training = True

