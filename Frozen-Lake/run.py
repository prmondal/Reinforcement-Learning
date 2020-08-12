import random
import numpy as np
import gym

random.seed(42)

total_episodes = 50000
target_avg_reward = 0.8

epsilon = 0.1
alpha = 0.01
gamma = 0.95

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    
    Q = np.zeros((state_space_size, action_space_size))

    total_rewards = []
    
    for e in range(total_episodes):
        episode_reward = 0
        steps = 0
        done = False

        current_state = env.reset()

        while not done:
            steps += 1

            if random.random() <= epsilon:
                action = random.randrange(action_space_size)
            else:
                action = np.argmax(Q[current_state])

            next_state, reward, done, _ = env.step(action)

            Q[current_state][action] = (1 - alpha) * Q[current_state][action] + alpha * (reward + gamma * np.amax(Q[next_state]))
            episode_reward += reward
            
            if done:
                total_rewards.append(episode_reward)
                break
            
            current_state = next_state

        if e % 500 == 0:
            avg_reward = 0
            
            for i in range(100):
                current_state= env.reset()
                done=False
                
                while not done: 
                    action = np.argmax(Q[current_state])
                    current_state, reward, done, info = env.step(action)
                    avg_reward += reward
            
            avg_reward = avg_reward/100
            
            print(f'Episode: {e+1}, Avg Reward: {avg_reward}')

            if avg_reward > target_avg_reward:
                print('Congratulations! Agent reached the goal.')
                np.savetxt('q-learned.txt', Q)
                break
        