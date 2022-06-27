import random
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam


class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(32, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 discount=0.99,
                 learning_rate=0.001,
                 epsilon=1.0,
                 decay=0.999,
                 epsilon_min=0.01,
                 batch_size=64,
                 train_start=1000,
                 memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        # 학습 관련 하이퍼 파라미터
        self.discount_factor = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        # replay memory
        self.memory = deque(maxlen=memory_size)
        # model
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        # target_model의 모든 가중치를 model의 가중치로 변경(복사)
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # experience를 replay memory에 저장
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # epsilon-greedy 알고리즘을 통해 행동 결정
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def train_model(self):
        # replay memory로부터 학습 데이터 가져오기
        mini_batch = random.sample(self.memory, self.batch_size)

        # 각 요소별로 분리
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # model에 사용된 파라미터들 불러오기
        model_params = self.model.trainable_variables

        # gradient 계산을 위해 변수 계산 순서 기억
        with tf.GradientTape() as tape:
            # 현재 기대 보상 계산
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 이후 기대 보상 계산
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts) # target_predicts에 대한 gradient 추적 중지
            max_q = np.amax(target_predicts, axis=-1)
            # target 값 계산
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            # loss 계산
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # gradient 계산
        grads = tape.gradient(loss, model_params)
        # gradient 반영
        self.optimizer.apply_gradients(zip(grads, model_params))


class DQNAgentPlayer:
    def __init__(self, filename, model_number, state_size, actions_num):
        self.state_size = state_size
        self.action_size = actions_num
        self.model = DQN(actions_num)
        if model_number == None:
            self.model.load_weights('{}/model'.format(filename))
        else:
            self.model.load_weights('{}/model-{}'.format(filename, model_number))

    def choose_action(self, state):
        state = state.reshape(1, -1)
        pred = self.model.predict(state)
        return np.argmax(pred), pred[0]