import json

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from collections import deque

import random

def shape_checker(array, shape):
    return
    assert array.shape == shape, \
        'SHAPE ERROR | array shape: ' + str(array.shape) + 'given shape: ' + str(shape)

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        self.fc_out_mu = Dense(
            self.action_dim,
            activation= 'tanh'
        )
        self.fc_out_std = Dense(
            self.action_dim,
            activation= 'softplus'
        )

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        mu = self.fc_out_mu(x)
        std = self.fc_out_std(x)

        mu = mu * self.action_bound
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        
        return mu, std

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1_state = Dense(64, activation= 'relu')
        self.fc1_action = Dense(64, activation= 'relu')
        self.fc2 = Dense(128, activation= 'relu')
        self.fc3 = Dense(128, activation= 'relu')
        self.fc_out = Dense(
            1
        )

    def call(self, x):
        h1 = self.fc1_state(x[0])
        h2 = self.fc1_action(x[1])
        h = concatenate([h1, h2], axis=-1)
        h = self.fc2(h)
        h = self.fc3(h)
        y = self.fc_out(h)

        return y

class DNN_SAC_Agent:
    def __init__(self, state_dim, action_dim, action_bound, args):
        
        self.render = False

        # env info
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # hyperparameters
        with open(args.param_path, 'r') as f:
            parameters = json.load(f)

        self.discount_factor = parameters['gamma']
        self.Actor_learning_rate = parameters['learning_rate']['actor']
        self.Critic_learning_rate = parameters['learning_rate']['critic']
        self.TAU = parameters['tau']
        self.temperature_param = parameters['temperature_param'] 
        self.BUFFER_SIZE = parameters['max_buffer_size']
        self.MIN_BUFFER_SIZE = parameters['min_buffer_size']
        self.BACTH_SIZE = parameters['batch_size']
        # buffer
        self.buffer = deque(maxlen=self.BUFFER_SIZE)

        # model
        self.Actor_model = Actor(self.action_dim, self.action_bound)
        self.Critic_model_1 = Critic() 
        self.Critic_model_2 = Critic()
        self.Critic_model_1_target = Critic()
        self.Critic_model_2_target = Critic()

        # optimizer
        self.Actor_optimizer = Adam(learning_rate= self.Actor_learning_rate)
        self.Critic_optimizer = Adam(learning_rate= self.Critic_learning_rate)

        # build
        state_in = Input((self.state_dim, ))
        action_in = Input((self.action_dim, ))
        self.Actor_model.build(input_shape= (None, state_dim))
        self.Critic_model_1([state_in, action_in])
        self.Critic_model_2([state_in, action_in])
        self.Critic_model_1_target([state_in, action_in])
        self.Critic_model_2_target([state_in, action_in])

        # test
        if not args.train:
            print("loading")
            self.Actor_model.load_weights(args.load_network_path)
            return

        self.update_target_model(
            self.Critic_model_1,
            self.Critic_model_1_target,
            1
        )        
        self.update_target_model(
            self.Critic_model_1,
            self.Critic_model_1_target,
            1
        )

        with open("./trained_model/tmp.txt", 'a') as f:
            f.write(
                "\n\n" + 
                "state_dim: " + str(self.state_dim) + '\n' +
                "action_dim: " + str(self.action_dim) + '\n' +
                "action_bound: " + str(self.action_bound) + '\n' +
                "parameters: " + args.param_path + '\n\n' +
                "episode: \n" +
                "reward: \n" 
                "=================================================" 
            )

        # summary
        self.Actor_model.summary()
        self.Critic_model_1.summary()
        print(
            'ENV INFO | ',
            'state_dim: ', self.state_dim,
            'action_dim: ', self.action_dim,
            'action_bound: ', self.action_bound
        )

        self.writer = tf.summary.create_file_writer('./summary/' + args.train_name)
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    def update_target_model(self, model, target_model, TAU):
        weights = model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = TAU * target_weights[i] + (1-TAU) * weights[i]  

        target_model.set_weights(target_weights)

    def get_action(self, state):
        mu, std = self.Actor_model(
            tf.convert_to_tensor(state)
        )
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        # shape_checker(mu, (self.BACTH_SIZE, 1))
        # shape_checker(std, (self.BACTH_SIZE, 1))
        # shape_checker(normal_prob, (self.BACTH_SIZE, 1))
        # shape_checker(action, (self.BACTH_SIZE, 1))
        return action

    def log_pdf(self, actions, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        log_pdf = normal_prob.log_prob(actions)
        actions = np.array(actions)
        shape_checker(actions, (self.BACTH_SIZE, 1))
        shape_checker(log_pdf, (self.BACTH_SIZE, 1))
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)
        
        return log_pdf
 
    # def sample_normal(self, mu, std):
    #     normal_prob = tfp.distributions.Normal(mu, std)
    #     action = normal_prob.sample()
    #     action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
    #     log_pdf = normal_prob.log_prob(action)
    #     log_pdf = tf.reduce_sum(log_pdf, 1, keepdims= True)

    #     return action, log_pdf

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

    def TD_target(self, rewards, next_states, dones):
        mu, std = self.Actor_model(
            tf.convert_to_tensor(next_states)
        )
        next_actions = self.get_action(next_states)
        log_pdfs = self.log_pdf(next_actions, mu, std)
        shape_checker(log_pdfs, (self.BACTH_SIZE, 1))
        next_Q1 = self.Critic_model_1(
            [
                tf.convert_to_tensor(next_states),
                tf.convert_to_tensor(next_actions)
            ]
        )
        next_Q2 = self.Critic_model_2(
            [
                tf.convert_to_tensor(next_states),
                tf.convert_to_tensor(next_actions)
            ]
        )
        next_Q = tf.minimum(next_Q1, next_Q2)

        next_Q = next_Q.numpy()
        rewards = np.reshape(rewards, [self.BACTH_SIZE, 1])
        dones = np.reshape(dones, [self.BACTH_SIZE, 1])
        shape_checker(next_Q, (self.BACTH_SIZE, 1))

        target = rewards + (1-dones) * self.discount_factor * (next_Q - self.temperature_param * log_pdfs)

        return target

    def critic_train(self, targets, states, actions):
        model_params = self.Critic_model_1.trainable_variables
        with tf.GradientTape() as tape:
            predict_Q = self.Critic_model_1(
                [
                    tf.convert_to_tensor(states),
                    tf.convert_to_tensor(actions)
                ],
                training = True      
            )
            loss = tf.reduce_mean(tf.square(targets - predict_Q))
            shape_checker(predict_Q, (self.BACTH_SIZE, 1))
            shape_checker(targets, (self.BACTH_SIZE, 1))
        grads = tape.gradient(loss, model_params)
        self.Critic_optimizer.apply_gradients(zip(grads, model_params))

        model_params = self.Critic_model_2.trainable_variables
        with tf.GradientTape() as tape:
            predict_Q = self.Critic_model_2(
                [
                    tf.convert_to_tensor(states),
                    tf.convert_to_tensor(actions)
                ],
                training = True      
            )
            loss = tf.reduce_mean(tf.square(targets - predict_Q))
            shape_checker(predict_Q, (self.BACTH_SIZE, 1))
            shape_checker(targets, (self.BACTH_SIZE, 1))
        grads = tape.gradient(loss, model_params)
        self.Critic_optimizer.apply_gradients(zip(grads, model_params))

    def actor_train(self, states):
        model_params = self.Actor_model.trainable_variables
        with tf.GradientTape() as tape:
            mu, std = self.Actor_model(
                tf.convert_to_tensor(states),
                training= True
            )
            actions = self.get_action(states)
            Q1 = self.Critic_model_1(
                [
                    tf.convert_to_tensor(states),
                    tf.convert_to_tensor(actions)
                ]
            )
            Q2 = self.Critic_model_2(
                [
                    tf.convert_to_tensor(states),
                    tf.convert_to_tensor(actions)
                ]
            )
            Q = tf.minimum(Q1, Q2)            
            log_pdf = self.log_pdf(actions, mu, std)
            loss = tf.reduce_mean(self.temperature_param * log_pdf - Q)
        grads = tape.gradient(loss, model_params)
        self.Actor_optimizer.apply_gradients(zip(grads, model_params))

    def train(self):

        if len(self.buffer) < self.MIN_BUFFER_SIZE:
            return

        batch = random.sample(self.buffer, self.BACTH_SIZE)

        states = [sample[0][0] for sample in batch]
        actions = [sample[1][0] for sample in batch]
        rewards = [sample[2][0] for sample in batch]
        next_states = [sample[3][0] for sample in batch]
        dones = [sample[4][0] for sample in batch]

        targets = self.TD_target(rewards, next_states, dones)

        self.critic_train(targets, states, actions)
        self.actor_train(states)

        self.update_target_model(
            self.Critic_model_1,
            self.Critic_model_1_target,
            self.TAU
        )        
        self.update_target_model(
            self.Critic_model_2,
            self.Critic_model_2_target,
            self.TAU
        )

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
