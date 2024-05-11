import gym
from gym.wrappers.record_video import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


# 신경망 정의
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        return self.neural_net(x.float())

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)
        x = torch.hstack((state, action))
        return self.neural_net(x.float())

# 리플레이 버퍼 정의
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), np.array(action), reward, np.array(next_state), done))

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(next_states).float(),
            torch.from_numpy(dones).int()
        )


# DDPG 에이전트 정의
class DDPGAgent:
    def __init__(self, state_size, action_size, action_low, action_high,
                 actor_lr=1e-4, critic_lr=1e-3, buffer_size=100000, batch_size=400,
                 gamma=0.99, tau=1e-3, noise_scale=0.5,
                 initial_std=1.0, min_std=0.1, decay_rate=0.01):

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

        # 액터 및 크리틱 네트워크 초기화
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_actor = Actor(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 최적화 및 버퍼
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        # 파라미터
        self.gamma = gamma
        self.tau = tau

        self.noise_scale = noise_scale
        self.current_std = initial_std
        self.min_std = min_std
        self.decay_rate = decay_rate

    def update_std(self, episode_num):
        self.current_std = max(self.min_std, self.current_std * np.exp(-self.decay_rate * episode_num))

    def act(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # state.shape: torch, [1,27]]
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()  # action.shape: np, [8,]
        self.actor.train()

        noise = self.noise_scale * (np.random.randn(self.action_size) * self.current_std)  # noise.shape: np, [8,]
        action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def learn(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()  # .shape: torch.float, [64,27][64,8][64,][64,27][64,]

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.unsqueeze(1).to(device)
        next_states = next_states.to(device)
        dones = dones.unsqueeze(1).to(device)

        # 타깃 액터와 크리틱의 계산
        next_actions = self.target_actor(next_states)  # [64,8]
        target_q_values = self.target_critic(next_states, next_actions)  # [64,1]
        q_targets = rewards + (self.gamma * target_q_values * (1 - dones))  # [64,1]

        # 크리틱 업데이트
        q_expected = self.critic(states, actions)  # [64,1]
        critic_loss = F.mse_loss(q_expected, q_targets)
        critic_log = critic_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 액터 업데이트
        predicted_actions = self.actor(states)  # [64,8]
        actor_loss = - self.critic(states, predicted_actions).mean()
        actor_log = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 타깃 네트워크 업데이트
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return (critic_log, actor_log)

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer.memory),
            'total_rewards': global_total_rewards,
            'Q_loss': mean_q_losses,
            'Policy_Loss': mean_policy_losses,
            'gamma': self.gamma,
            'tau': self.tau
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        global global_total_rewards
        global mean_q_losses
        global mean_policy_losses
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.replay_buffer.memory = deque(checkpoint['replay_buffer'], maxlen=100000)
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        global_total_rewards = checkpoint['total_rewards']
        mean_q_losses = checkpoint['Q_loss']
        mean_policy_losses = checkpoint['Policy_Loss']

if __name__ == '__main__':

    # 환경 및 DDPG 에이전트 초기화
    env_name = 'Ant-v4'
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, './video', episode_trigger=lambda episode_number: (episode_number + 1) % 33 == 0)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    agent = DDPGAgent(state_size, action_size, action_low, action_high,
                      actor_lr=1e-4, critic_lr=1e-3, buffer_size=100000, batch_size=64,
                      gamma=0.99, tau=1e-3, noise_scale=0.1,
                      initial_std=1.0, min_std=0.1, decay_rate=0.01
                      )

    # 학습 루프
    num_episodes = 50
    checkpoint_path = './ddpg_checkpoint.pth'
    save_interval = 33

    global_total_rewards = []
    q_losses = []
    policy_losses = []
    mean_q_losses = []
    mean_policy_losses = []

    # 필요한 경우 이전에 저장된 체크포인트에서 상태를 불러옴
    try:
        agent.load_checkpoint(checkpoint_path)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print("No checkpoint found or failed to load, starting fresh.")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        q_losses = []
        policy_losses = []

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 학습
            log = agent.learn()
            if log is not None:
                q_loss, policy_loss = log
                q_losses.append(q_loss)
                policy_losses.append(policy_loss)

        global_total_rewards.append(total_reward)
        mean_q_losses.append(np.array(q_losses).mean())
        mean_policy_losses.append(np.array(policy_losses).mean())

        agent.update_std(episode)  # 각 에피소드마다 표준 편차 갱신
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        # 일정한 에피소드마다 체크포인트를 저장
        if (episode + 1) % save_interval == 0:
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}.")

    # 환경 종료
    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(mean_q_losses, label='Q-Loss')
    plt.plot(mean_policy_losses, label='Policy Loss')
    plt.plot(global_total_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses & Rewards Over Time')
    plt.legend()
    plt.show()