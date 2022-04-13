from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import argparse
from DNN_PPO_agent import DNN_PPO_Agent
from CNN_PPO_agent import CNN_PPO_Agent
from CV_feature_extractor import *
import numpy as np
from datetime import datetime

def shape_check(array, shape):
    assert array.shape == shape, \
        'shape error | array.shape ' + str(array.shape) + ' shape: ' + str(shape)

def main(args):
    # 환경 정의 및 설정 
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(args.env_path, 
                           worker_id=np.random.randint(65535),
                           side_channels=[engine_configuration_channel])
    env.reset()

    # agent
    action_dim = 3 # xyz Quart...
    action_bound = 1 # max_input 
    if args.case_num ==  0:
        state_dim = 9 # not defined
        agent = DNN_PPO_Agent(state_dim, action_dim, action_bound, args)
    elif args.case_num ==  1:
        state_dim = 3 # not defined
        agent = DNN_PPO_Agent(state_dim, action_dim, action_bound, args)
    elif args.case_num == 2:
        state_dim = (84, 84, 3) # not defined
        agent = CNN_PPO_Agent(state_dim, action_dim, action_bound, args)
    else:
        raise Exception("check case_num you trying (sould be in range[0:2]")

    # behavior 이름 불러오기 및 timescale 설정
    behavior_name = list(env.behavior_specs)[0]
    engine_configuration_channel.set_configuration_parameters(time_scale=args.time_scale)

    score_list = []
    max_score = 1e-9
    EPISODE = 100000

    # 전체 진행을 위한 반복문 (10 에피소드 반복)
    for e in range(EPISODE):
        # 환경 초기화 
        env.reset()

        # decision_steps와 terminal_steps 정의
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # state
        state = decision_steps.obs[0][0] # need to check
        if args.case_num == 1:
            state = extract(state)
        state = np.reshape(state, [1, state_dim])

        # 파라미터 초기화 
        score, step, done = 0, 0, 0

        # 에피소드 진행을 위한 while문 
        while not done:
            step += 1

            # get action
            action = agent.get_action(state)
            action = np.clip(action, -action_bound, action_bound)
            action = np.reshape(action, [1, action_dim])

            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)

            env.set_actions(behavior_name, action_tuple)

            # 행동 수행 
            env.step()

            # 행동 수행 후 에이전트의 정보 (상태, 보상, 종료 여부) 취득
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # done, reward check
            done = len(terminal_steps.agent_id)>0
            reward = terminal_steps.reward[0] if done else decision_steps.reward[0]

            # next_state
            if done:
                next_state = terminal_steps.obs[0][0]
            else:
                next_state = decision_steps.obs[0][0]

            if args.case_num == 1:
                next_state = extract(next_state)

            # reshaping            
            action = np.reshape(action, [1, action_dim])
            next_state = np.reshape(next_state, [1, state_dim])
            reward = np.reshape(reward, [1, 1])
            done = np.reshape(done , [1, 1])

            # train
            if args.train:
                agent.sample_append(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
                agent.train()

            # for next_step
            score += reward 
            state = next_state
        
        # score store
        score = score[0][0]
        score_list.append(score)

        if args.train:
            # report train result
            print(
                'EPISODE: ', e+1,
                'STEP: ', step,
                'SCORE: ', round(score, 3),
            )
            agent.draw_tensorboard(score, step, e)

            # update weights
            if score > max_score:
                max_score = score
                now = datetime.now()
                now = now.strftime('%m%d%H%M')
                save_name = './trained_model/' + args.train_name + '/' +  now + 'EPISODE' + str(e+1)

                agent.Actor_model.save_weights(save_name)

    # 환경 종료 
    env.close() 

if __name__=='__main__':
    now = datetime.now()
    now = now.strftime('%m%d%H%M')

    parser = argparse.ArgumentParser(description="choice options")
    parser.add_argument('--train_name', type=str, 
                        default=now, dest="train_name", action="store",
                        help='trained model would be saved in that directory')
    parser.add_argument('--env_path', type=str, 
                        default='./envs/env1/env1.x86_64', dest="env_path", action="store",
                        help='environment path e.g. --env_path \'./envs/...\'')
    parser.add_argument('--case_num', type=int,
                        default=0, dest='case_num', action='store',
                        help='0: observation vector, 1: cv feature extractor, 2: CNN extractor')
    parser.add_argument('--train',
                        default=False, dest='train', action='store_true',
                        help='add this arg to train')
    parser.add_argument('--load_network', type=str,
                        default='NOT_LOADED', dest='load_network_path', action='store',
                        help='to load trained network add path e.g. --load_network \'./...\'')
    parser.add_argument('--time_scale', type=float, 
                        default=20.0, dest='time_scale', action="store",
                        help='to accellerate simul (consider you PC spec)')
    parser.add_argument('--parameters', type=str, 
                        default='./config/parameters.json', dest='param_path', action="store",
                        help='NN parameters')
    
    args = parser.parse_args()

    if args.train:
        with open("./trained_model/tmp.txt", 'a') as f:
            f.write(
                "\nTRAIN INFO\t" + now + '\n' +
                "train name: " + str(args.train_name) + '\n' + 
                "env_path: " + str(args.env_path) + '\n' + 
                "case_name: " + str(args.case_num) + '\n' + 
                "load network: " + str(args.load_network_path)
            )    

    main(args)
