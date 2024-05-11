import gym
from gym.wrappers.record_video import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import random
from collections import deque


# 신경망 정의
class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=256, epsilon=0.0, noise_rate=0.1):
        super(Actor, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()
        )
        self.epsilon = epsilon
        self.noise_rate = noise_rate

    def mu(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        return self.neural_net(x.float())

    def forward(self, x):
        mu = self.mu(x)
        action = mu + torch.normal(0, self.epsilon, mu.size(), device=mu.device) * self.noise_rate
        return action


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
        input_vector = torch.hstack((state, action))
        return self.neural_net(input_vector.float())


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
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.BoolTensor(dones)
        )


# DDPG 에이전트 정의
class DDPGAgent:
    def __init__(self, state_size, action_size, action_low, action_high):
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(100000, 64)

        # 파라미터
        self.gamma = 0.99
        self.tau = 1e-3

    def act(self, state, noise_scale=0.1):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()
        self.actor.train()

        noise = noise_scale * np.random.randn(self.action_size)
        action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def learn(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # 타깃 액터와 크리틱의 계산
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + (self.gamma * target_q_values.squeeze() * (1 - dones))

        # 크리틱 업데이트
        q_expected = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 액터 업데이트
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 타깃 네트워크 업데이트
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer.memory),
            'gamma': self.gamma,
            'tau': self.tau
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
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


if __name__ == '__main__':

    # 환경 및 DDPG 에이전트 초기화
    env_name = 'Ant-v4'
    env = gym.make(env_name, render_mode="rgb_array")
    # env = gym.make(env_name, batch_size=num_envs, episode_length=episode_length)
    env = RecordVideo(env, './video', episode_trigger=lambda episode_number: (episode_number + 1) % 30 == 1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    agent = DDPGAgent(state_size, action_size, action_low, action_high)

    # 학습 루프
    num_episodes = 10000
    checkpoint_path = './ddpg_checkpoint.pth'
    save_interval = 30

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

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 학습
            agent.learn()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        # 일정한 에피소드마다 체크포인트를 저장
        if (episode + 1) % save_interval == 0:
            agent.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}.")

    # 환경 종료 및 녹화된 비디오 표시
    env.close()
    # show_video()
