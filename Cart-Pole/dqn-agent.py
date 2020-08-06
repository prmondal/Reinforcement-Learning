import random
import numpy as np
import tensorflow
from collections import deque
from tensorflow.keras import Sequential, layers, optimizers
import gym
import sys

EPISODES = 1000
MODEL_WEIGHTS_PATH = './dqn-agent-trained-model-24-24.h5'

class DQNAgent:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.score_threshold = 475

        self.print_model_summary = False
        self.pretrained = True

        self.memory = deque(maxlen=2000)
        self.init_model()
    
    def damp_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.damp_epsilon()

    def init_model(self):
        self.model = Sequential()
        self.model.add(layers.Input(shape=self.state_space_size))
        self.model.add(layers.Dense(24, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(layers.Dense(24, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(layers.Dense(self.action_space_size, kernel_initializer='he_uniform'))
        
        self.model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate), metrics='mse')

        if self.print_model_summary:
            self.model.summary()
        
        if self.pretrained:
            self.model.load_weights(MODEL_WEIGHTS_PATH)
        
    def choose(self, state):
        if random.random() <=  self.epsilon and not self.pretrained:
            return random.randrange(self.action_space_size)
        
        pred = self.model.predict(state)
        return np.argmax(pred[0])
    
    def reshape_state(self, state):
        return np.reshape(state, (1, self.state_space_size))

    def learn(self):
        batch_size = min(len(self.memory), self.batch_size)
        training_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in training_batch:
            target = self.model.predict(state)

            if not done:
                predict_future_reward = self.model.predict(next_state)
                target[0][action] = reward + self.discount_factor * np.amax(predict_future_reward[0])
            else:
                target[0][action] = reward
            
            self.model.fit(state, target, epochs=1, verbose=0)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    scores = []

    for e in range(EPISODES):
        state = env.reset()
        state = agent.reshape_state(state)
        done = False
        score = 0
        steps = 0

        while not done:
            steps = steps + 1
            env.render()
            
            action = agent.choose(state)

            next_state, reward, done, _ = env.step(action)
            next_state = agent.reshape_state(next_state)
            
            agent.memorize(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                print(f'Episode: {e+1}, Reward: {steps}, Epsilon: {agent.epsilon}')

                scores.append(score)

                # avg score in last 10 episodes
                mean_score = np.mean(scores[-min(10, len(scores)):])
                if mean_score >= agent.score_threshold:
                    agent.model.save_weights(MODEL_WEIGHTS_PATH)
                    sys.exit()

                break
        
        if not agent.pretrained:
            agent.learn()
    
    agent.model.save_weights(MODEL_WEIGHTS_PATH)