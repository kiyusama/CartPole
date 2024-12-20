import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random
import math
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    
def get_action(state):
    global steps_done
    sample = random.random()
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
    
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

def train():
    if len(memory) < BATCH_SIZE:
        return
    sample_experiences = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*sample_experiences))

    #終了時はnext_stateがないので除外する
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #すべて行動のQ値を計算された後,gather(dim, index)で実際に選択した行動のみ取り出す
    action_values = policy_net(state_batch).gather(1, action_batch)

    next_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():#計算効率化のため伝播の追跡を切る
        next_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_values = (next_values * GAMMA) + reward_batch

    #予測Q値のactionValuesと目標Q値(仮定)のexpectedValuesの差を最小化していく
    criterion = nn.MSELoss()
    loss = criterion(action_values, expected_values.unsqueeze(1))

    #policyNetで学習
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)#勾配爆発を防ぐ。勾配を-100~100に制限
    optimizer.step()

# ハイパーパラメータ
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
env = gym.make('CartPole-v1')
state, info = env.reset()
input_size = len(state)
output_size = env.action_space.n
policy_net = DQN(input_size, output_size)
target_net = DQN(input_size, output_size)
target_net.load_state_dict(policy_net.state_dict())#ターゲットにポリシーネットの重みをコピー
memory = ReplayMemory(10000)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
steps_done = 0
num_episodes = 500
episode_rewards = [] # エピソードごとの報酬を格納するリスト

for episode in range(num_episodes):
    total_reward = 0
    done = False
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    while not done:
        action = get_action(state)
        observation, reward, done, truncated, info = env.step(action.item())
        reward = torch.tensor([reward])
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        train()
        total_reward += reward.item()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict) #targetにコピー

        if total_reward >= 10000:
            torch.save(policy_net.state_dict(), 'PyTorch/CartPole2/elite_models/cartpole_dqn.pth')
            break
        
    torch.save(policy_net.state_dict(), 'PyTorch/CartPole2/models/cartpole_dqn.pth')
    print(f'Episode {episode} : total reward: {total_reward}')
    episode_rewards.append(total_reward)

plt.plot(episode_rewards)
plt.title('Episode Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.xlim(0, num_episodes)
plt.ylim(0, 2000)
plt.show()