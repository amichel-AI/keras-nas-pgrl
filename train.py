import numpy as np
import argparse
import importlib
import keras as keras
from keras.datasets import cifar10, mnist
from net_reflection import NetReflection
from actions import ActionHelper
from reinforce import NASenv, PGAgent

def parse_args():
    thisDescription = "Keras-NASNet-train: A Keras implementation of Neural Architecture Search (NAS) with Reinforcement Learning' (Policy Gradient approach)"
    parser = argparse.ArgumentParser(description=thisDescription)

    parser.add_argument('--path', default=".")
    parser.add_argument('--num_layers', default=3)

    args = parser.parse_args()
    args.num_layers = int(args.num_layers)
    return args

def load_data():
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    nb_classes = 10

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    (x_train, y_train), (x_test, y_test) = preprocess_mnist(x_train, y_train, x_test, y_test)
    return (x_train, y_train), (x_test, y_test), nb_classes

def preprocess_mnist(x_train, y_train, x_test, y_test):
    # add singleton channel
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    return (x_train, y_train), (x_test, y_test)

def preprocess_cifar10(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])
    return (x_train, y_train), (x_test, y_test)

def train(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train,
              batch_size=NASconfig.batch_size,
              epochs=max(1,int(NASconfig.max_epochs)),
              callbacks=model.callbacks,
              #validation_data=(x_test,y_test),
              shuffle=True,
              verbose=0)
    return history.history['loss'][-1], history.history['acc'][-1]

def test(model, x_test, y_test):
    return model.evaluate(x_test, y_test, batch_size=NASconfig.batch_size, verbose=0)

def main():
    global args
    global NASconfig

    args = parse_args()
    NASconfig = getattr(importlib.import_module('config', args.path + '/'), 'Config')

    (x_train, y_train), (x_test, y_test), nb_classes = load_data()
    env = NASenv(NASconfig, (x_train, y_train), (x_test, y_test), nb_classes, train_callback=train, test_callback=test)

    agent = PGAgent(env)
    agent.learn(max_episodes=NASconfig.max_num_episodes, batch_size=NASconfig.batch_size)

    # Save model and weights
    #if not os.path.isdir(save_dir):
    #    os.makedirs(save_dir)
    #model_path = os.path.join(save_dir, model_name)
    #model.save(model_path)
    #print('Saved trained model at %s ' % model_path)

if __name__ == '__main__':
  main()