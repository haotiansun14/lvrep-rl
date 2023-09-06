import collections
import numpy as np
import torch
import torch.distributions as pyd


Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'reward', 'next_state', 'done']
	)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)
	
	def retrieve_all(self):
		return Batch(
			state=torch.FloatTensor(self.state[:self.size]).to(self.device),
			action=torch.FloatTensor(self.action[:self.size]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[:self.size]).to(self.device),
			reward=torch.FloatTensor(self.reward[:self.size]).to(self.device),
			done=torch.FloatTensor(self.done[:self.size]).to(self.device),
		)

class D_base(object):
	def __init__(self, state_dim, K=int(1e4)):
		self.K = K
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		mu_s = torch.zeros(state_dim).to(self.device)
		cov_s = torch.eye(state_dim).to(self.device) 
		self.base_measure = pyd.MultivariateNormal(loc=mu_s, covariance_matrix=cov_s)
		
		# sampling
		self.state = self.base_measure.sample((self.K,)).to(self.device)
		self.prob = self.base_measure.log_prob(self.state).to(self.device)

	def base_prob(self, state):
		return self.base_measure.log_prob(state).exp().to(self.device)

	def state_and_prob(self):
		return self.state, self.prob

