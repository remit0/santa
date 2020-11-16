from collections import namedtuple
import random
from nets import NeuralNet
from torch import optim
import torch
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:

    def __init__(self, layers, lr):
        self.net = NeuralNet(layers).float()
        self.target_net = NeuralNet(layers).float()
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def predict(self, states, use_target):
        predictor = self.target_net if use_target else self.net
        states = torch.from_numpy(states).float()
        predictor.eval()
        with torch.no_grad():
            action_values = predictor(states)
        return action_values

    def sample_action(self, state, eps):
        if np.random.uniform(0, 1) < eps:
            return np.random.choice(100)
        else:
            state = np.expand_dims(state, axis=0)
            return np.argmax(self.predict(state, use_target=False)).item()

    def one_hot_encode(self, actions):
        encoded = torch.zeros(len(actions), 100)
        for i, action in enumerate(actions):
            encoded[i][action] = 1
        return encoded

    def cost(self, pred_action_values, actions, targets):
        action_values = torch.sum(pred_action_values * self.one_hot_encode(actions), dim=1)
        targets = torch.from_numpy(targets).float()
        cost = torch.sum(torch.square(targets - action_values))
        return cost

    def update(self, memory, batch_size, gamma):
        transitions = memory.sample(batch_size)

        states = []
        rewards = []
        next_states = []
        dones = []
        actions = []
        for transition in transitions:
            states.append(transition.state)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            actions.append(transition.action)
            dones.append(transition.done)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        dones = np.array(dones)

        next_q = np.max(self.predict(next_states, use_target=True).numpy(), axis=1)
        targets = rewards + gamma * next_q * np.invert(dones).astype(np.float32)

        self.net.train()
        self.optimizer.zero_grad()
        states = torch.from_numpy(states).float()
        pred_action_values = self.net(states)
        cost = self.cost(pred_action_values, actions, targets)
        cost.backward()
        self.optimizer.step()
