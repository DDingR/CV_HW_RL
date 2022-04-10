from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import sys
from PPO_agent import Agent
import numpy as np
from datetime import datetime
import cv2

def shape_check(array, shape):
    assert array.shape == shape, \
        'shape error | array.shape ' + str(array.shape) + ' shape: ' + str(shape)

def main():
    if len(sys.argv) == 1:
        file_name = './CarDrive.x86_64'
    else:
        file_name = sys.argv[1]
    # 환경 정의 및 설정 
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name, 
                           worker_id=np.random.randint(65535),
                           side_channels=[engine_configuration_channel])
    env.reset()
    print("after reset")
    # agent
    state_dim = 9 # need to check
    action_dim = 2 # xz
    action_bound = 1 # max_input 
    agent = Agent(state_dim, action_dim, action_bound)

    # behavior 이름 불러오기 및 timescale 설정
    behavior_name = list(env.behavior_specs)[0]
    engine_configuration_channel.set_configuration_parameters(time_scale=20.0)

    score_list = []
    max_score = 1e-9
    EPISODE = 100000

    # 전체 진행을 위한 반복문 (10 에피소드 반복)
    for e in range(EPISODE):
        # 환경 초기화 
        env.reset()

        # decision_steps와 terminal_steps 정의
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        print(np.array(decision_steps.obs[0]).shape)
        print(np.array(decision_steps.obs[1]).shape)
        print(np.array(decision_steps.obs[0][0]).shape)        

        img = decision_steps.obs[0][0]
        cv2.imshow('image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()      

        # state
        state = decision_steps.obs[0][0] # need to check
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
            
            done = len(terminal_steps.agent_id)>0
            reward = terminal_steps.reward[0] if done else decision_steps.reward[0]

            if done:
                # next_state = [terminal_steps.obs[i][0] for i in range(6)]
                next_state = terminal_steps.obs[0][0]
            else:
                # next_state = [decision_steps.obs[i][0] for i in range(6)]
                next_state = decision_steps.obs[0][0]
            # 매 스텝 보상을 에피소드에 대한 누적보상에 더해줌 
            
            action = np.reshape(action, [1, action_dim])
            next_state = np.reshape(next_state, [1, state_dim])
            reward = np.reshape(reward, [1, 1])
            done = np.reshape(done , [1, 1])

            agent.sample_append(
                state,
                action,
                reward,
                next_state,
                done
            )

            score += reward 
            state = next_state
            # print(state)
            # print(action)
        
        score = score[0][0]
        score_list.append(score)

        print(
            'EPISODE: ', e+1,
            'STEP: ', step,
            'SCORE: ', round(score, 3),
        )

        agent.draw_tensorboard(score, step, e)

        if score > max_score:
            max_score = score
            now = datetime.now()
            now = now.strftime('%m%d%H%M')
            agent.Actor_model.save_weights('DDPG' + now)

    # 환경 종료 
    env.close() 

if __name__=='__main__':
    main()
