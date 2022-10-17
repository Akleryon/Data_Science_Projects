import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym import wrappers



import experience_replay, image_preprocessing

class Brain(nn.Module):

    def __init__(self, nbr_actions):
        super(Brain, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size= 5)
        self.convolution2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 2)
        self.fc1 = nn.Linear(in_features= self.count_neurons((1, 80, 80)), out_features= 40)
        self.fc2 = nn.Linear(in_features= 40, out_features= nbr_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.data.view(1, -1).size(1)
        return x

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Body(nn.Module):

    def __init__(self, T):
        super(Body, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
    

class ai(nn.Module):

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):

        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()  

doom_env = image_preprocessing.PreprocessImage(gym.make('VizdoomCorridor-v0'), width = 256, height = 256, grayscale = True)
doom_env = wrappers.Monitor(doom_env, "videos", force = True)

numbers_actions = doom_env.action_space.n

network = Brain(numbers_actions)
body = Body(T = 1.0)

AI = ai(brain = network, body = body)

n_steps = experience_replay.NStepProgress(env = doom_env, ai = AI, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = network(input)
        sum_rewards = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            sum_rewards = step.reward + gamma * sum_rewards
        state = series[0].state
        target = output[0].data
        target[series[0].action] = sum_rewards
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

class MA():

    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)

ma = MA(100)

loss = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        prediction = network(inputs)
        loss_error = loss(prediction, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, average reward: %s" %(str(epoch, str(avg_reward))) )




