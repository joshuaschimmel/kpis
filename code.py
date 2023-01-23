# !pip install wandb
# !wandb login

### Import

!pip install stable-baselines3
!pip install keras-rl

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
import google.colab.drive as gdrive
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
gdrive.mount('/content/drive')

import gym
import numpy as np
import bisect
import matplotlib.pyplot as plt
from stable_baselines3.dqn import DQN
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import os.path
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

# import wandb

### CartPole Environment 




### Random Search

def random_search(episodes_num):
  """Runs cartpole test with random actions.
  """
  env = gym.make("CartPole-v1")
  episode_scores = []
  episode_counter = 0
  
  for episode in range(episodes_num):
    episode_counter += 1
    score = 0
    done = False
    current_observation = env.reset()

    while not done and score < 200:
      # select random sample from action space
      action = env.action_space.sample()
      next_observation, reward, done, info = env.step(action)
      score += reward
      current_observation = next_observation

    # episode finished -> cleanup
    # save score
    episode_scores.append(score)

    # mark first episode that keeps the pole balanced for the entire episode
    if score >= 200:
      break
      

  # test run cleanup, close environment and return metrics
  env.close()
  return episode_scores, episode_counter

### Random Search with weights
As seen here https://github.com/kvfrans/openai-cartpole

def random_search_weights(episodes_num):
  """Runs cartpole environment with random weights per episode.
  """
  env = gym.make("CartPole-v1")
  episode_scores = []
  episode_counter = 0
  
  for episode in range(episodes_num):
    episode_counter += 1
    score = 0
    done = False
    current_observation = env.reset()
    action_policy = np.random.rand(4) * 2 -1
    # done = True , score =10
    while not done and score < 500:
      # select random sample from action space
      action = 0 if np.matmul(action_policy, current_observation) < 0 else 1
      next_observation, reward, done, info = env.step(action)
      score += reward
      current_observation = next_observation
      #print(score)
    # episode finished -> cleanup
    # save score
    episode_scores.append(score)

    # mark first episode that keeps the pole balanced for the entire episode
    if score >= 200:
      break

  # test run cleanup, close environment and return metrics
  env.close()
  return episode_scores, episode_counter

def CartPole_Random_Search(runs = 200, episodes = 1000):
  rewards = []
  counters = []
  for i in range(runs):
    episode_rewards, first_successful_episode = random_search(episodes_num=episodes)
    print("Run: ", i, first_successful_episode)
    rewards.append(episode_rewards)
    counters.append(first_successful_episode)

  return rewards, counters

def CartPole_Random_Weights_Search(runs = 200, episodes = 1000):
  rewards = []
  counters = []
  for i in range(runs):
    episode_rewards, first_successful_episode = random_search_weights(episodes_num=episodes)
    print("Run: ", i, " Needed episodes: ", first_successful_episode)
    rewards.append(episode_rewards)
    counters.append(first_successful_episode)

  return rewards, counters

def plot_durations(durations, episodes, title, label):
  """Creates histogramm of durations data.
  """
  bins = np.arange(0, 100, 10)
  plt.title(title)
  plt.hist(durations, bins=bins, label=label, alpha=0.75)
  plt.xlabel("Required episodes to first reach a reward of 200")
  plt.ylabel("Frequency")
  #plt.savefig("/content/drive/MyDrive/" + label + "-speed.png")

def plot_rewards(rewards, title, label):
  plt.title(title)
  plt.plot(rewards, alpha=0.75)
  plt.xlabel("Epsiodes")
  plt.ylabel("Average Reward")
  plt.show()

def eval_random_search(runs = 1000, episodes = 10000):
  """Evaluates the random algorithms
  """  
  #rewards, durations = CartPole_Random_Search(runs, episodes)
  rewards_weights, durations_weights = CartPole_Random_Weights_Search(runs, episodes)
  np.save('/content/drive/MyDrive/kpis-graphs/random-1.npy', durations_weights)
  #plot_durations(durations, episodes, "Random Search Speed", "random")
  #print(np.sum(durations_weights)/10000.0)
  #plot_durations(durations_weights, episodes, "Random Policy Search Speed", "policy")
  
  # aggregated_rewards = np.average(np.array(rewards).reshape(runs, -1, 50), axis=2)
  # average_rewards = np.average(aggregated_rewards, axis=0)
  # plot_rewards(average_rewards, "Average Reward per episode for Random Search", "random")

  # aggregated_rewards_weights = np.average(np.array(rewards_weights).reshape(runs, -1, 50), axis=2)
  # average_rewards_weights = np.average(aggregated_rewards_weights, axis=0)
  # plot_rewards(average_rewards_weights, "Average Reward per episode for Random Policy Search", "policy")
eval_random_search(runs=100, episodes=1000)

r, d = CartPole_Random_Search(100, 1000)
np.save('/content/drive/MyDrive/kpis-graphs/control-1.npy', d)







### Q-Learning and SARSA Functions

def initialization(observation_num = 5,action_num = 2):
  value_function = np.random.uniform(low=--1,high=1,size=(observation_num,observation_num,observation_num,observation_num,action_num))
  discrete_observation_space =  {"Cart Position": np.linspace(-2.4,2.4,observation_num),
                                 "Cart Velocity": np.linspace(-5,5,observation_num),
                                 "Pole Angle": np.linspace(-0.2095,0.2095,observation_num),
                                 "Pole Angular Velocity": np.linspace(-5,5,observation_num)}
  return value_function,discrete_observation_space

def continuous2discrete(continuous_observation,discrete_observation_space):
  discrete_observation = []
  for i,key in enumerate(discrete_observation_space.keys()):
    discrete_observation.append(bisect.bisect(discrete_observation_space[key],continuous_observation[i])-1)
  return discrete_observation

def epsilon_greedy(env,epsilon,value_function,current_observation):
  explore_or_exploit = np.random.choice(a=["explore","exploit"],p=[epsilon,1-epsilon])
  if explore_or_exploit == "explore":
    return env.action_space.sample()
  else:
    return np.argmax(value_function[current_observation])

### Q-Learning

def q_learning(gamma,alpha,epsilon,episodes_num):
  env = gym.make("CartPole-v1")
  value_function,discrete_observation_space = initialization()
  max_score = 0
  episode_scores = []
  episode_counter = 0
  for episode in range(episodes_num):
    episode_counter += 1
    score = 0
    done = False
    current_observation_continuous = env.reset()
    current_observation = tuple(continuous2discrete(current_observation_continuous,discrete_observation_space))

    while not done and score < 200:
      action = epsilon_greedy(env,epsilon,value_function,current_observation)
      next_observation_continuous, reward, done, info = env.step(action)
      score += reward
      next_observation = tuple(continuous2discrete(next_observation_continuous,discrete_observation_space))
      value_function[current_observation][action] += alpha*(reward + gamma*np.max(value_function[next_observation]) - value_function[current_observation][action])
      current_observation = next_observation
    
    episode_scores.append(score)

    if score >= 200:
      break
    
  env.close()
  return episode_scores, episode_counter

def CartPole_Q_Learning(runs= 200, episodes = 10000):
  rewards = []
  counters = []
  for i in range(runs):
    episode_rewards, first_successful_episode = q_learning(gamma=0.95,alpha=0.1,epsilon=0.1,episodes_num=episodes)
    print("Run: ", i, "Counter: ", first_successful_episode)
    rewards.append(episode_rewards)
    counters.append(first_successful_episode)
  return rewards, counters

# s, results = CartPole_Q_Learning(runs=4)
# plt.hist(results, 50, facecolor='g', alpha=0.75)
# plt.xlabel("Required episodes to reach 200")
# plt.ylabel("Frequency")
# #plt.savefig('/content/drive/MyDrive/KPIS-Seminar-DQN-Model/test.png')
# print(np.sum(results)/1000.0)

def eval_q_learning(runs = 1000, episodes = 10000):
  """Evaluates the random algorithms
  """    
  rewards, counters = CartPole_Q_Learning(runs, episodes)
  np.save('/content/drive/MyDrive/kpis-graphs/q-learning-1.npy', counters)
  #print(np.sum(counters)/1000.0)

  #plot_durations(counters, episodes, "Q-Learning Speed", "Q")
  
  #aggregated_rewards = np.average(np.array(rewards).reshape(runs, -1, 50), axis=2)
  #average_rewards = np.average(np.array(rewards), axis=0)
  #plot_rewards(average_rewards, "Average Reward per episode for Q-Learning", "Q")
eval_q_learning(runs=100, episodes=1000)





### SARSA

def sarsa(gamma,alpha,epsilon,episodes_num):
  env = gym.make("CartPole-v1")
  value_function,discrete_observation_space = initialization()
  max_score = 0
  scores = []
  counter = 0 
  for episode in range(episodes_num):
    counter += 1
    score = 0
    done = False
    current_observation_continuous = env.reset()
    current_observation = tuple(continuous2discrete(current_observation_continuous,discrete_observation_space))
    current_step = 0
    action = epsilon_greedy(env, epsilon,value_function,current_observation)

    while not done and current_step < 200:
      #env.render()
      next_observation_continuous, reward, done, info = env.step(action)
      score += reward
      next_observation = tuple(continuous2discrete(next_observation_continuous,discrete_observation_space))
      next_action = epsilon_greedy(env, epsilon,value_function,next_observation)
      value_function[current_observation][action] += alpha*(reward + gamma*value_function[next_observation][next_action] - value_function[current_observation][action])
      current_observation = next_observation
      action = next_action
      current_step += 1
      
    scores.append(score)
    
    if max_score < score:
      max_score = score
    if score >= 200:
      break 
    
  env.close()
  return scores, counter





def CartPole_Sarsa(runs = 1000, episodes = 10000):
  #wandb.init(project="Q-Learning", entity="ml-experiments")
  results = []
  for i in range(runs):
    scores, counter = sarsa(gamma=0.95,alpha=0.1,epsilon=0.1,episodes_num=episodes)
    print("Run: ", i, "Counter: ", counter)
    results.append(counter)
  return results

def eval_sarsa(runs = 1000, episodes = 10000):
  """Evaluates the random algorithms
  """    
  counters = CartPole_Sarsa(runs, episodes)
  #print(np.sum(counters)/1000.0)
  
  np.save('/content/drive/MyDrive/kpis-graphs/sarsa-1.npy', counters)

  #plot_durations(counters, episodes, "SARSA Speed", "sarsa")
  
  #aggregated_rewards = np.average(np.array(rewards).reshape(runs, -1, 50), axis=2)
  #average_rewards = np.average(np.array(rewards), axis=0)
  #plot_rewards(average_rewards, "Average Reward per episode for Q-Learning", "Q")
eval_sarsa(runs=100, episodes=1000)

# results = CartPole_Sarsa()
# plt.hist(results, 50, facecolor='g', alpha=0.75)
# plt.xlabel("Required episodes to reach 200")
# plt.ylabel("Frequency")
# plt.show()
# print(np.sum(results)/1000.0)

### DQN (stable_baselines3)

def train_model(env, timesteps,lr,gamma):
  callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1000, verbose=1)
  model = DQN(policy="MlpPolicy", env=env, learning_rate=lr, gamma=gamma)
  model.learn(total_timesteps=timesteps, callback=callback_max_episodes)  
  #model.save('/content/drive/MyDrive/KPIS-Seminar-DQN-Model/stable_baselines3_dqn_cartpole')
  return model

def stable_baselines3_dqn(timesteps,gamma,lr,episodes_num):
  env = gym.make("CartPole-v1")
  if os.path.isfile('/content/drive/MyDrive/KPIS-Seminar-DQN-Model/stable_baselines3_dqn_cartpole.zip'):
    print("Loading the model")
    model = DQN.load('/content/drive/MyDrive/KPIS-Seminar-DQN-Model/stable_baselines3_dqn_cartpole')
  else:
    print("Training model")
    model = train_model(env,timesteps,lr,gamma)
  
  print("Loading finished")
  max_score = 0
  scores = []
  counter = 0 
  for episode in range(episodes_num):
    counter += 1
    score = 0
    done = False
    current_step = 0
    current_observation = env.reset()

    while not done and current_step < 200:
      #env.render()
      action, used_state = model.predict(current_observation, deterministic=False)
      next_observation, reward, done, info = env.step(action)
      score += reward
      current_observation = next_observation
      current_step += 1
    
    scores.append(score)
    
    if max_score < score:
      max_score = score
      if score >= 200: 
        break 
    
  env.close()
  return counter

def CartPole_DQN():
  #wandb.init(project="Q-Learning", entity="ml-experiments")
  results = []
  for i in range(1000):
    counter = stable_baselines3_dqn(timesteps=300000,gamma=0.95,lr=0.0001,episodes_num=10000)
    print("Run: ", i, "Counter: ", counter)
    results.append(counter)
  return results

# results = CartPole_DQN()
# plt.hist(results, 50, facecolor='g', alpha=0.75)
# plt.xlabel("Required episodes to reach 200")
# plt.ylabel("Frequency")
# plt.show()
# print(np.sum(results)/1000.0)

#stable_baselines3_dqn(timesteps=300000,gamma=0.95,lr=0.0001,episodes_num=3000)

#plot_durations(results, 1000, "DQN Speed", "dqn")

log_dir = "./drive/MyDrive/kpis-logs/dqn"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)


callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10000, verbose=1)
model = DQN(policy="MlpPolicy", env=env, learning_rate=0.1, gamma=0.95)
model.learn(total_timesteps=200000, log_interval=1, callback=[callback_max_episodes])


#load_results(log_dir)
np.amax(env.get_episode_lengths())
#x, y = ts2xy(, "timesteps")


log_dir = "./drive/MyDrive/kpis-logs/dqn"
os.makedirs(log_dir, exist_ok=True)
counter = []
for i in range(50):
  env = gym.make("CartPole-v1")
  env.reset()
  env = Monitor(env, log_dir)
  callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10000, verbose=1)
  model = DQN(policy="MlpPolicy", env=env, learning_rate=0.01, gamma=0.95)
  model.learn(total_timesteps=200000, log_interval=1, callback=[callback_max_episodes])
  first_episode = 10000
  amax = np.amax(env.get_episode_lengths())
  if amax >= 200:
    first_episode = np.argmax(env.get_episode_lengths())
  counter.append(first_episode)
  print("i: ", i, " counter: ", first_episode, " amax: ", amax)

np.save("/content/drive/MyDrive/kpis-graphs/dqn-1.npy")

np.save("/content/drive/MyDrive/kpis-graphs/dqn-1.npy", counter)

### Deep Sarsa from this example: [keras-rl/examples/sarsa_cartpole.py](https://github.com/keras-rl/keras-rl/blob/master/examples/sarsa_cartpole.py)



# Example from https://github.com/keras-rl/keras-rl/blob/master/examples/sarsa_cartpole.py

def buil_model(env):
  
  nb_actions = env.action_space.n

  # Next, we build a very simple model.
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  print(model.summary())

  policy = EpsGreedyQPolicy()
  sarsa = SARSAAgent(model=model, policy=policy, gamma=0.95, nb_actions=nb_actions, nb_steps_warmup=10)
  sarsa.compile(Adam(learning_rate=1e-3), metrics=['mae'])

  return sarsa

# Get the environment and extract the number of actions.
env = gym.make('CartPole-v1')
env.seed(42)
sarsa = buil_model(env)

training_history = sarsa.fit(env, nb_steps=10000, nb_max_episode_steps=200, visualize=False, verbose=1)
sarsa.save_weights('/content/drive/MyDrive/KPIS-Seminar-Sarsa-Model/sarsa.hdf5', overwrite=True)


sarsa.save_weights('/content/drive/MyDrive/KPIS-Seminar-Sarsa-Model/sarsa.hdf5', overwrite=True)

env = gym.make('CartPole-v1')
env.seed(42)
model = buil_model(env)

if os.path.isfile('/content/drive/MyDrive/KPIS-Seminar-Sarsa-Model/sarsa.hdf5'):
  print('found model')
  model.load_weights('/content/drive/MyDrive/KPIS-Seminar-Sarsa-Model/sarsa.hdf5')

test_data = model.test(env, nb_episodes=1000, nb_max_episode_steps=200, visualize=False)

counter = test_data.history['nb_steps']

plot_durations(counter, 1000, "DSN Speed", "dsn")


print(np.sum(counter)/1000.0)

def eval_dsn(runs = 1000, episodes = 10000):
  """Evaluates the random algorithms
  """    
  #counters = CartPole_Sarsa(runs, episodes)
  results = []
  for i in range(runs):
    #counter = stable_baselines3_dqn(timesteps=300000,gamma=0.95,lr=0.0001,episodes_num=episodes)
    res = model.test(env, nb_episodes=episodes, nb_max_episode_steps=200, visualize=False)
    counter = np.argmax(res.history['nb_steps'])
    print("Run: ", i, "Counter: ", counter)
    results.append(counter)
  return results
  print(np.sum(counters)/1000.0)

  plot_durations(counters, episodes, "DSN Speed", "dsn")
  
  
eval_dsn(runs=1000, episodes=200)

