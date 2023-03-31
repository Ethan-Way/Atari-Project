import random
from collections import deque
import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# coded using github copilot

class DQNAgent:
    def __init__(self):
        self.env = gym.make('Alien-v0', render_mode='human')
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1] *\
                     self.env.observation_space.shape[2]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, episodes, batch_size=64):
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, episodes, time, self.epsilon))
                    break
            if len(self.memory) > batch_size:
                self.replay(batch_size)
        self.save("Alien-dqn.h5")

    def test(self, episodes):
        self.load("Alien-dqn.h5")
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                if done:
                    print("episode: {}"
                          .format(e))
                    break

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        return model


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train(episodes=1000)
    agent.test(episodes=100)

