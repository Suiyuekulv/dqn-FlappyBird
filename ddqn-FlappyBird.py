import torch
from torch import nn
import cv2
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque


ACTIONS = 2
GAMMA = 0.99
MIN_REPLAY_SIZE = 3200
# MIN_REPLAY_SIZE = 100
EPSILON_DECAY = 50000  # frames over which to anneal epsilon /30000
FINAL_EPSILON = 0.01  # /0.0001
INITIAL_EPSILON = 0.9

PROB_DECAY = 100000
FINAL_PROB = 0.5
INITIAL_PROB = 0.95

REPLAY_MEMORY = 30000  # /30000
BATCH_SIZE = 32  # 32
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 2000  # /3000
PATH = 'logs_ddqn/model.pt'

step_list = [0.0]
rew_buffer = [0.0]
result1 = deque([0.0])  # average reward over every 10 episodes
result2 = deque([0.0])  # average episode step over every 10 episodes


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2))
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTIONS)
        )

    def forward(self, x):
        # return self.net(x)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def act(self, state):
        obs_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.forward(obs_t)
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


def update_state(frame, state_t):
    frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    if torch.equal(state_t, -1 * torch.ones(1, 4, 80, 80)):
        state_t1 = np.stack((frame, frame, frame, frame), axis=1)
        state_t1 = state_t1.reshape((1, 4, 80, 80))
        state_t1 = torch.as_tensor(state_t1, dtype=torch.float32)
    else:
        frame = np.reshape(frame, (1, 1, 80, 80))
        frame = torch.from_numpy(frame)

        state_t1 = torch.cat([frame, state_t[:, :3, :, :]], dim=1)

    return state_t1


def train_dqn(game_state, arg):
    # global step_list, rew_buffer, result1, result2

    # instantiate two CNN
    online_net = CNN()
    target_net = CNN()

    # load weights of online net to target net
    target_net.load_state_dict(online_net.state_dict())
    # allocate optimizer
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    # loss_fn = nn.SmoothL1Loss(reduce=False, size_average=False)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    # define replay buffer
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # print(do_nothing)
    frame_t, r_t, terminal = game_state.frame_step(do_nothing)
    state_t = update_state(frame_t, -1 * torch.ones(1, 4, 80, 80))

    # fill in replay buffer
    if arg == 'start':
        for _ in range(MIN_REPLAY_SIZE):
            action_t = np.zeros([ACTIONS])

            if random.random() <= 0.9:
                action_index = 0
            else:
                action_index = 1
            action_t[action_index] = 1

            frame_t1, r_t, terminal = game_state.frame_step(action_t)
            state_t1 = update_state(frame_t1, state_t)
            replay_buffer.append((state_t, action_t, r_t, state_t1, terminal))
            state_t = state_t1
        epoch = 1

    elif arg == 'resume':
        # load general checkpoint
        checkpoint = torch.load(PATH)
        online_net.load_state_dict(checkpoint['online_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_buffer = checkpoint['replay_buffer']
        epoch = checkpoint['epoch']+1
        # # load results
        # result_data = torch.load('logs_ddqn/result.pt')
        # result1 = result_data['result1']
        # result2 = result_data['result2']

    # define variable to trace history
    print('Minimal buffer size reached!')
    episode = 1
    episode_reward = 0.0
    episode_step = 0.0

    # main training loop
    while 1:
        epsilon = np.interp(epoch, [0, EPSILON_DECAY], [INITIAL_EPSILON, FINAL_EPSILON])
        prob = np.interp(epoch, [0, PROB_DECAY], [INITIAL_PROB, FINAL_PROB])

        action_t = np.zeros([ACTIONS])
        # take actions epsilon-greedily

        if random.random() <= epsilon:
            if random.random() <= prob:
                action_index = 0
            else:
                action_index = 1
            action_t[action_index] = 1
        else:
            action_index = online_net.act(state_t)
            action_t[action_index] = 1

        # one step forward
        frame_t1, r_t, terminal = game_state.frame_step(action_t)
        state_t1 = update_state(frame_t1, state_t)
        replay_buffer.append((state_t, action_t, r_t, state_t1, terminal))
        state_t = state_t1

        # record the history
        episode_reward += r_t
        episode_step += 1
        if terminal:
            episode += 1
            step_list.append(episode_step)
            rew_buffer.append(episode_reward)
            # if episode % 11 == 0:
            #
            #     step_list.pop(0)
            #     result1.append(np.average(step_list))
            #
            #     step_list = [0.0]
            #
            #     rew_buffer.pop(0)
            #     result2.append(np.average(rew_buffer))
            #
            #     rew_buffer = [0.0]

            episode_reward = 0.0
            episode_step = 0.0

        # gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        actions = np.array([t[1] for t in transitions])
        rews = np.array([t[2] for t in transitions])
        dones = np.array([t[4] for t in transitions])

        obses_t = torch.cat([t[0] for t in transitions])
        new_obses_t = torch.cat([t[3] for t in transitions])
        actions_t = torch.from_numpy(actions)
        actions_t = torch.as_tensor(actions_t, dtype=torch.int64)
        rews_t = torch.from_numpy(rews).unsqueeze(-1)
        rews_t = torch.as_tensor(rews_t, dtype=torch.float32)
        dones_t = torch.from_numpy(dones).unsqueeze(-1)
        dones_t = torch.as_tensor(dones_t, dtype=torch.int64)

        # Compute Targets
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values
        # Compute Loss
        q_values = online_net(obses_t)

        action_q_values = torch.sum(q_values*actions_t, dim=1)

        targets = torch.reshape(targets, (-1,))
        loss = loss_fn(action_q_values, targets)

        # Gradient Descent
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        # Update Target Network
        if epoch % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # save progress every 10000 iterations
        if epoch % 10000 == 0:
            print(f'Epoch = {epoch}')
            torch.save({
                'epoch': epoch,
                'online_net_state_dict': online_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'replay_buffer': replay_buffer,
            }, PATH)

        # terminate training after 200000 epochs and return results
        if epoch % 80000 == 0:
            # torch.save({
            #     'result1': result1,
            #     'result2': result2,
            # }, 'logs_ddqn/result.pt')
            return

        epoch += 1

def greedy_playing(game_state):
    # instantiate a CNN
    online_net = CNN()
    # load general checkpoint
    checkpoint = torch.load(PATH)
    online_net.load_state_dict(checkpoint['online_net_state_dict'])

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # print(do_nothing)
    frame_t, r_t, terminal = game_state.frame_step(do_nothing)
    state_t = update_state(frame_t, -1 * torch.ones(1, 4, 80, 80))

    while(1):
        # act greedily
        action_index = online_net.act(state_t)
        action_t = np.zeros([ACTIONS])
        action_t[action_index] = 1

        # one step forward
        frame_t1, r_t, terminal = game_state.frame_step(action_t)
        state_t1 = update_state(frame_t1, state_t)
        state_t = state_t1

# open up a game state to communicate with emulator
game_state = game.GameState()

# greedy_playing(game_state)


train_dqn(game_state, 'resume')

# training result visualization
# result1.popleft()
# result2.popleft()
#
# plt.figure()
# plt.plot(result1)
# plt.title('Average reward over every 10 episodes')
# plt.xlabel('Episode Set(/10)')
# plt.ylabel('Average Reward')
# plt.show()
#
# plt.figure()
# plt.plot(result2)
# plt.title('Average episode step over every 10 episodes')
# plt.xlabel('Episode Set(/10)')
# plt.ylabel('Average episode step')
# plt.show()



