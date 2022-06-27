import random
import matplotlib.pyplot as plt
from env import CartPoleEnv
from dqn import DQNAgent, DQNAgentPlayer
from concatvideo import concatVideos


def env_test():
    env = CartPoleEnv()
    state = env.reset()

    for i in range(100):
        env.render()
        action = random.randint(0, 1)
        next_state, reward, done = env.step(action)
        print(state, action, reward, done)


def train_loop(filename):
    # 환경 불러오기
    env = CartPoleEnv()
    state_size = env.state_size
    actions_num = env.actions_num

    # 에이전트 생성
    agent = DQNAgent(state_size, actions_num)

    # 각 episode 당 총 프레임 수 저장
    scores = []
    # 학습 종료조건을 위해 평균 score 계산
    score_avg = 0

    num_episode = 300 # 최대 episode
    for episode in range(num_episode):
        done = False
        score = 0

        # 환경 초기화
        state = env.reset()
        state = state.reshape(1, -1)
        action = None

        while not done and score <= 475:
            # 현재 상태 표시
            env.render(episode=episode, action=action, recording=True)

            # 행동 결정
            action = agent.choose_action(state)

            # 행동을 환경에 반영
            next_state, reward, done = env.step(action)
            next_state = next_state.reshape(1, -1)

            # 경험 저장
            agent.remember(state, action, reward, next_state, done)

            # 해당 에피소드의 총 스텝 수 저장
            score += 1

            # experience가 충분히 쌓아면 학습
            if len(agent.memory) >= agent.train_start:
                # step이 지남에 따라 epsilon 감소
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                agent.train_model()

            # 다음 state로 업데이트
            state = next_state

        # target 모델 갱신
        agent.update_target_model()

        # score 이동 평균
        score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
        print('episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f}'
              .format(episode, score_avg, len(agent.memory), agent.epsilon))

        scores.append(score_avg)

        fig, ax = plt.subplots()
        ax.plot(range(0, episode+1), scores, 'b')
        ax.set_xlabel('episode')
        ax.set_ylabel('average score')
        plt.close(fig)
        fig.savefig(filename + '_graph.png')

        # 25 에피소드당 모델 저장
        if episode % 25 == 0:
            agent.model.save_weights('./{}/model-{}'.format(filename, episode), save_format='tf')

        # 평균 스코어가 400을 넘어가거나 최대 에피소드에 도달하면 학습 루프 종료
        if score_avg > 400 or episode == num_episode - 1:
            agent.model.save_weights('./{}/model'.format(filename), save_format='tf')
            break

        env.save("tmp\episode{:03d}.mp4".format(episode))

    concatVideos("{}_training".format(filename), "tmp\\")


def replay_loop(filename, episode_num=None):
    env = CartPoleEnv()
    state_size = env.state_size
    actions_num = env.actions_num

    agent = DQNAgentPlayer(filename, episode_num, state_size, actions_num)
    done = False
    action = None
    state = env.reset()
    #env.cart_x = -env.x_threadhold / 2.0
    #env.pole_theta = env.theta_threadhold - 0.01
    #state[0] = -env.x_threadhold / 2.0
    #state[2] = env.theta_threadhold - 0.01
    score = 0

    while not done and score <= 475:
        env.render(action=action, recording=False)
        action, pred = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        print(state, pred, action, done)
        state = next_state
        score += 1

    #env.save("{}_replay.mp4".format(filename))

    print("Score:", score)


if __name__ == '__main__':
    filename = "20220605_layer3_128_64_32"
    # env_test()
    train_loop(filename)
    replay_loop(filename)
    # concatVideos("result", "tmp\\")

