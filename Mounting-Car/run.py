# Solved in around 5k episodes

import random
import numpy as np
import gym

random.seed(42)

total_episodes = 10000
target_position = 0.5
target_avg_reward = -110

epsilon = 1.0
epsilon_decay = 0.99
alpha = 0.5
gamma = 0.95

position_indices_map = dict()
velocity_indices_map = dict()

idx = 0
for i in np.arange(-1.2, 0.7, 0.1):
    position_indices_map[round(i, 1)] = idx
    idx += 1

idx = 0
for i in np.arange(-0.07, 0.07, 0.01):
    velocity_indices_map[round(i, 2)] = idx
    idx += 1

def get_index(Q, state):
    pos = round(state[0], 1)
    vel = round(state[1], 2)

    return position_indices_map[pos] * len(velocity_indices_map) + velocity_indices_map[vel]

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    
    Q = np.zeros((len(position_indices_map) * len(velocity_indices_map), action_space_size))

    total_rewards = []
    
    for e in range(total_episodes):
        episode_reward = 0
        steps = 0
        done = False
        reached_target = False

        current_state = env.reset()

        while not done:
            env.render()
            steps += 1

            curr_table_idx = get_index(Q, current_state)

            if random.random() <= epsilon:
                action = random.randrange(action_space_size)
            else:
                action = np.argmax(Q[curr_table_idx])
            
            next_state, reward, done, _ = env.step(action)
            next_table_idx = get_index(Q, next_state)

            Q[curr_table_idx][action] = (1 - alpha) * Q[curr_table_idx][action] + alpha * (reward + gamma * np.amax(Q[next_table_idx]))
            episode_reward += reward

            if done:
                if steps < 200:
                    print(f'Success in {steps} steps.')

                if next_state[0] >= target_position:
                    reached_target = True

                total_rewards.append(episode_reward)
                break
            
            current_state = next_state
        
        epsilon *= epsilon_decay

        if np.mean(total_rewards[-min(len(total_rewards), 100)]) >= target_avg_reward:
            print('Hurray! Congratulations! Agent reached the summit.')
            np.savetxt('q-learned.txt', Q)
            env.close()
            break

        if e == 0 or (e+1) % 50 == 0:
            print(f'Episode: {e+1}, Reward: {episode_reward}, Target Reached: {reached_target}, Epsilon: {epsilon}')