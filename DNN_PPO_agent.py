import json

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
# import matplotlib.pyplot as plt
import random

def shape_check(array, shape):
    assert array.shape == shape, \
        'shape error | array.shape ' + str(array.shape) + ' shape: ' + str(shape)

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound, state_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.state_dim = state_dim

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        # self.fc4 = Dense(512, activation='relu')
        self.fc_mu = Dense(
            self.action_dim, 
            activation='tanh', # [-1, 1]
            kernel_initializer=RandomUniform(-1e-3, 1e-3)
        )
        self.fc_std = Dense(
            self.action_dim, 
            activation='softplus',
            kernel_initializer=RandomUniform(-1e-3, 1e-3)
        )

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        mu = Lambda(lambda x: x*self.action_bound)(mu)
        return mu, std

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        # self.fc4 = Dense(512, activation='relu')
        self.fc_out = Dense(
            1, # value 하나만 output 이기 때문에
            kernel_initializer=RandomUniform(-1e-3, 1e-3)
        )

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
        y = self.fc_out(x)
        return y

class DNN_PPO_Agent:
    def __init__(self, state_dim, action_dim, action_bound, args):

        self.render = False

        # env parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]

        # hyperparameters
        with open("./parameter.json", 'r') as f:
            parameters = json.load(f)

        self.Actor_learning_rate = parameters['learning_rate']['actor']
        self.Critic_learning_rate = parameters['learning_rate']['critic']
        self.gamma = parameters['gamma']
        self.RATIO_CLIPPING = parameters['clip_ratio'] # clip coefficient
        self.MAX_BUFFER_SIZE = parameters['max_buffer_size']
        self.batch_size = parameters['batch_size']
        self.EPOCH = parameters['epoch']
        self.GAE_param = parameters['GAE_param']
        # buffer
        self.buffer = []

        # model define
        self.Actor_model = Actor(self.action_dim, self.action_bound, self.state_dim)
        self.Critic_model = Critic(self.state_dim)
        self.Actor_model.build(input_shape=(None, self.state_dim))
        self.Critic_model.build(input_shape=(None, self.state_dim))
        self.Actor_optimizer = Adam(
            learning_rate= self.Actor_learning_rate, 
            # clipnorm=1.0
        )    
        self.Critic_optimizer = Adam(
            learning_rate= self.Critic_learning_rate,
            # clipnorm=1.0
        )

        with open("./trained_model/train_history.txt", 'a') as f:
            f.write(
                "\n\n" + 
                "state_dim: " + str(self.state_dim) + '\n' +
                "action_dim: " + str(self.state_dim) + '\n' +
                "action_bound: " + str(self.state_dim) + '\n' +
                "=================================================" 
            )

        # summary
        self.Actor_model.summary()
        self.Critic_model.summary()

        print(
            'ENV INFO | ',
            'state_dim: ', self.state_dim,
            'action_dim: ', self.action_dim,
            'action_boud: ', self.action_bound
        )

        self.writer = tf.summary.create_file_writer('./summary/' + args.train_name)
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu, std = self.Actor_model(
            tf.convert_to_tensor(state)
        )
        mu = mu.numpy()[0]
        std = std.numpy()[0]
        clipped_std = np.clip(std, self.std_bound[0], self.std_bound[1])
        # print(self.action_dim, mu,  std)
        action = np.random.normal(mu, clipped_std, size = self.action_dim)
        # print(mu, std,clipped_std, action)
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action
    
    def sample_append(self, state, action, reward, next_state, done):
        self.buffer.append(
            [
                state,
                action,
                reward,
                next_state,
                done
            ]
        )

    def GAE_target(self, state_list, reward_list, next_state_list, done_list):
        value = self.Critic_model(
            tf.convert_to_tensor(state_list)
        )
        next_value = self.Critic_model(
            tf.convert_to_tensor(next_state_list)
        )
        reward_list = np.reshape(reward_list, [self.batch_size, 1])
        done_list = np.reshape(done_list, [self.batch_size, 1])
        value = value.numpy()
        next_value = next_value.numpy()

        done = done_list[self.batch_size-1][0]
        # next_value = next_value[self.MAX_BUFFER_SIZE-1][0]
        # print(done)
        if done:
            # print("0s")
            next_value[self.MAX_BUFFER_SIZE-1][0] = 0

        delta_list = reward_list + self.gamma * next_value - value
        delta_list = np.flip(delta_list)
        
        # print(delta_list.shape, next_value.shape, value.shape)

        GAE = []
        for i, delta in enumerate(delta_list):
            if i == 0:
                GAE.append(delta)
            else:
                GAE.append(delta + self.gamma * self.GAE_param * GAE[i-1])
        GAE = np.flip(GAE)
        GAE = np.reshape(GAE, [self.batch_size, 1])
        
        target = GAE + value
        # print(GAE.shape, value.shape)
        return GAE, target

    def GAE_target_test(self, state_list, reward_list, next_state, done):
        predict = self.Critic_model(
            tf.convert_to_tensor(state_list)
        )
        predict = predict.numpy()
        # shape_check(predict, (self.batch_size, 1))
        next_value = 0
        next_GAE = 0

        if not done:
            next_value = self.Critic_model(
                tf.convert_to_tensor(
                    np.reshape(next_state, [1, self.state_dim])
                )
            )
            next_value = next_value.numpy()
            # shape_check(next_value, (1, 1))
        
        reward_list = np.reshape(reward_list, [self.batch_size, 1])
        # shape_check(reward_list, (self.batch_size, 1))
        delta = np.zeros_like(reward_list)
        GAE = np.zeros_like(reward_list)

        for i in reversed(range(0, self.batch_size)):
            delta[i] = reward_list[i] + self.gamma * next_value - predict[i]
            GAE[i] = delta[i] + self.gamma * self.GAE_param * next_GAE
            next_value = predict[i]
            next_GAE = GAE[i]

        target = GAE + predict
        # shape_check(target, (self.batch_size, 1))
        return GAE, target

    def Critic_train(self, target_list, state_list):
        model_params = self.Critic_model.trainable_variables
        with tf.GradientTape() as tape:
            predict = self.Critic_model(
                tf.convert_to_tensor(state_list)
            )
            advantage = target_list - predict
            loss = tf.reduce_mean(tf.square(advantage))
            # print(target_list.shape, predict.shape, advantage.shape)
        grads = tape.gradient(loss, model_params)
        self.Critic_optimizer.apply_gradients(zip(grads, model_params))

    def Actor_train(self, state_list, action_list, old_log_pdf, GAE):
        model_params = self.Actor_model.trainable_variables
        with tf.GradientTape() as tape:
            mu, std = self.Actor_model(
                tf.convert_to_tensor(state_list)
            )
            log_pdf = self.log_pdf(mu, std, action_list)
            ratio = tf.exp(log_pdf - old_log_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio * GAE, clipped_ratio * GAE)
            loss = tf.reduce_mean(surrogate)
            # print(surrogate.shape, loss)
        grads = tape.gradient(loss, model_params)
        self.Actor_optimizer.apply_gradients(zip(grads, model_params))            

    def train(self):

        if len(self.buffer) < self.MAX_BUFFER_SIZE:
            return
        
        for _ in range(self.EPOCH):
            # print("train!")  
            batch = random.sample(self.buffer, self.batch_size) 
            state_list = [sample[0][0] for sample in batch]
            action_list = [sample[1][0] for sample in batch]
            reward_list = [sample[2][0] for sample in batch]
            next_state_list = [sample[3][0] for sample in batch]
            done_list = [sample[4][0] for sample in batch]

            # GAE, target = self.GAE_target(state_list, reward_list, next_state_list, done_list)
            GAE, target = self.GAE_target_test(state_list, reward_list, next_state_list[self.batch_size-1], done_list[self.batch_size-1])

            mu, std = self.Actor_model(
                tf.convert_to_tensor(state_list)
            )
            old_log_pdf = self.log_pdf(mu, std, action_list)        

            self.Critic_train(target, state_list)
            self.Actor_train(state_list, action_list, old_log_pdf, GAE)
            
        self.buffer.clear()

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
