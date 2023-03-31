import gym
import numpy as np


def get_reward(prev_lives, lives, score, prev_score, done):
    reward = 0
    if not done:
        # Reward for getting a score
        if score > prev_score:
            reward += 10
        # Penalty for losing a life
        if lives < prev_lives:
            reward -= 10
        # living penalty
        if score == prev_score and score > 0:
            reward -= 0.01
    else:
        # Penalty for losing the game
        reward -= 100
    return reward


class QLearningAgent:
    def __init__(self, env_name, learning_rate=0.999, discount_factor=0.95,
                 num_episodes=10, epsilon=1.0,
                 max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01, steps=100):

        self.env = gym.make(env_name, render_mode='human')
        num_states = self.env.observation_space.shape[0] * self.env.observation_space.shape[1] *\
                     self.env.observation_space.shape[2]
        num_actions = self.env.action_space.n
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.steps = steps

    def choose_action(self, state):
        # determine if a random action should be taken or the best action
        random_action = np.random.uniform()
        if random_action > self.epsilon:
            action = np.argmax(self.q_table[state, :])
        else:
            action = self.env.action_space.sample()
        return action

    def update_q_table(self, state, action, new_state, prev_lives, lives, prev_score, score, done):
        reward = get_reward(prev_lives, lives, score, prev_score, done)
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * \
                                      (reward + self.discount_factor * np.max(self.q_table[new_state, :]) -
                                       self.q_table[state, action])

    def train(self):
        for episode in range(self.num_episodes):
            print("Starting episode:", episode)
            state = self.env.reset()
            state = np.array(state).reshape(1, self.q_table.shape[0])
            total_reward = 0
            done = False
            prev_lives = self.env.env.ale.lives()
            prev_score = 0

            while not done:  # use if game does have terminal state
                # for step in range(self.steps): # use if the game does not have a terminal state
                # Choose an action
                action = self.choose_action(state)

                # Take the chosen action and save the new state and reward
                action = np.clip(action, 0, self.env.action_space.n - 1)
                new_state, reward, done, info = self.env.step(action)

                new_state = np.array(new_state).reshape(1, self.q_table.shape[0])
                lives = self.env.env.ale.lives()
                score = info['episode_frame_number']
                self.update_q_table(state, action, new_state, prev_lives, lives, prev_score, score, done)

                total_reward += reward
                state = new_state
                prev_lives = lives
                prev_score = score

            print("Score:", total_reward)

            # Decay epsilon
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

        self.env.close()
        print("Training complete!")


agent = QLearningAgent(env_name="Alien-v0", num_episodes=100000, steps=1000, discount_factor=.3, decay_rate=.001)
agent.train()
