import numpy as np
import torch
# import gym
import gymnasium as gym
import argparse
import os

from datetime import datetime

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.vlsac import ctrl_agent, vlsac_agent, speder_agent

eps_greedy = 0.01

import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
	
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", default=0, type=int)                     
  parser.add_argument("--alg", default="ctrlsac")                     # Alg name (sac, vlsac, spedersac, ctrlsac)
  parser.add_argument("--env", default="HalfCheetah-v4")          # Environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=float)# 25e3 Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # 5e3 How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=float)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--hidden_dim", default=256, type=int)      # Network hidden dims
  parser.add_argument("--feature_dim", default=256, type=int)      # Latent feature dim
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--learn_bonus", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--extra_feature_steps", default=3, type=int)
  args = parser.parse_args()

  env = gym.make(args.env)
  eval_env = gym.make(args.env)
  env.reset(seed=args.seed)
  eval_env.reset(seed=args.seed)
  max_length = env._max_episode_steps
  print(f"Max length: {max_length}")
  # setup log 
  log_path = f'log/{args.env}/{args.alg}/{args.seed}/{datetime.now().strftime("%D-%H:%M")}'
  summary_writer = SummaryWriter(log_path)


  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # 
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0] 
  max_action = float(env.action_space.high[0])

  print(f"Algo: {args.alg}. Env: {args.env}. Seed: {args.seed}.")
  print(f"State dim: {env.observation_space}. Action dim: {env.action_space}. Max action: {max_action}. Reward range: {env.reward_range}, Spec: {env.spec}, Metadata: {env.metadata}")

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "action_space": env.action_space,
    "discount": args.discount,
    "tau": args.tau,
    "hidden_dim": args.hidden_dim,
  }

  # Initialize policy
  if args.alg == "sac":
    agent = sac_agent.SACAgent(**kwargs)
  elif args.alg == 'vlsac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = args.feature_dim
    agent = vlsac_agent.VLSACAgent(**kwargs)
  elif args.alg == 'spedersac':
    kwargs['extra_feature_steps'] = args.extra_feature_steps
    kwargs['feature_dim'] = 1024  
    kwargs['hidden_dim'] = 1024
    agent = speder_agent.SPEDER_SACAgent(**kwargs)
  elif args.alg == 'ctrlsac':
    kwargs['extra_feature_steps'] = True
    kwargs['feature_dim'] = 2048 # 2048  
    kwargs['hidden_dim'] = 1024
    agent = ctrl_agent.CTRL_SACAgent(**kwargs)
  
  print(f"kwargs: {kwargs}")

  replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)

  # Evaluate untrained policy
  evaluations = [util.eval_policy(agent, eval_env)]

  state, done = env.reset()[0], False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  timer = util.Timer()

  print(f"Start training. Alg: {args.alg}. Env: {args.env}. Seed: {args.seed}.")
  for t in range(int(args.max_timesteps)):
    # print(f"Step {t+1}.")
    episode_timesteps += 1

    # Select action randomly or according to policy
    if t < args.start_timesteps:
      action = env.action_space.sample()
    else:
      # epsilon greedy
      if np.random.uniform(0, 1) < eps_greedy:
        action = env.action_space.sample()
      else:
        action = agent.select_action(state, explore=True)

    # Perform action
    # next_state, reward, done, _ = env.step(action) 
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    done_bool = float(done) if episode_timesteps < max_length else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward
    
    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
      info = agent.train(replay_buffer, batch_size=args.batch_size) 
      # info = agent.train(replay_buffer, batch_size=512) 

    if done: 
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
      # Reset environment
      state, done = env.reset()[0], False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1 

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      steps_per_sec = timer.steps_per_sec(t+1)
      evaluation = util.eval_policy(agent, eval_env)
      evaluations.append(evaluation)

      if t >= args.start_timesteps:
        info['evaluation'] = evaluation
        for key, value in info.items():
          summary_writer.add_scalar(f'info/{key}', value, t+1)
        summary_writer.flush()

      print('Step {}. Steps per sec: {:.4g}.'.format(t+1, steps_per_sec))

  summary_writer.close()

  print('Total time cost {:.4g}s.'.format(timer.time_cost()))
