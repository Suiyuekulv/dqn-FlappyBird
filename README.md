# Playing FlappyBird with double deep Q-learning

Play the game 'Flappy Bird' using double deep Q-learning. The deep neural networks are implemented by PyTorch.

## Introduction
This project creates an intelligent agent that automatically learns how to play the game 'Flappy Bird', by following a double deep Q-learning scheme, which is described in _Deep Reinforcement Learning with Double Q-learning_[1]. By introducing the technique of experience replay, the algorithm performs well in learning stability. 

## Dependencies
- Python 3.9
- PyTorch 1.10.1
- pygame
- OpenCV

## Setup
_This program is to be run in Anaconda virtual environment._ Open Anaconda Prompt and go to a prefered directory for the following setups:
 
1. Install [PyTorch](https://pytorch.org/)

   _CUDA 11.3_
    ```
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
   _CPU_
    ```
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

2. Install pygame and OpenCV
   ```
   python3 -m pip install -U pygame --user
   conda install -c conda-forge opencv
   ```

3. Clone the repository
   ```
   git clone https://github.com/yifanyin11/double-deep-Q-learning-FlappyBird.git
   ```

4. Run the program
   ```
   cd double-deep-Q-learning-FlappyBird
   python ddqn_FlappyBird.py
   ```

## Functions
_**update_state(frame, state_t)**_

Convert current frame returned by the game emulator to a binary image, and stack it with the last three binary frames to form the new state.

&nbsp;

_**train_dqn(game_state, arg)**_

Core function of this program. 

Two training modes are available, controlled by _arg_. If _arg_=='start', training will restart. It will first go through an observation stage without updating networks. The agent will take random actions in this stage and store state transitions in Replay Buffer, until _MIN_REPLAY_SIZE_ is reached. Then the function enters main training loop to iteratively updates action value function, as approximated by an online network, by following the reinforcement learning algorithm described in [1]. The weights of the online network will be copied to a target network every _TARGET_UPDATE_FREQ_ epochs to reduce the correlation between steps of target calculation. Network updates are performed by the built-in differentiation engine of PyTorch, _torch.autograd_, and the model will be saved every 10000 iterations.

If _arg_=='resume', training will be resumed from the last saved checkpoint. The greedy policy of the current online network will be used to fill the buffer up to _MIN_REPLAY_SIZE_. Training will continue after that.

&nbsp;

_**greedy_playing(game_state)**_

Play the game under greedy policy, as given by current online network read from the last saved checkpoint.


## Reference

[1] Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

[2] Maim Lapan. Deep Reinforcement Learning Hands-On. Second Edition. Published by Packt Publishing Ltd.

## Disclaimer

The game emulator and some variable names are from the following repository:

[yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
