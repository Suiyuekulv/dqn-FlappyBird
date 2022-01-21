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
 
1. Install Torch from [PyTorch](https://pytorch.org/)

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
_update_state(frame, state_t)_

Convert current frame returned by the game emulator to a binary image, and stack it with the last three binary frames to form the new state.
  
  
_train_dqn(game_state, arg)_

Core function in this program. 

Two training modes are available, controlled by argument _arg_. If _arg_=='start', training will restart. It will first go through an observation stage without updating networks. The agent will take random actions in this stage and store state transitions in the replay buffer, until _MIN_REPLAY_SIZE_ is reached. Then the function enters main training loop to iteratively updates action value function with an online network and a target network, by following the reinforcement learning scheme described in [1]. The weights of the online network will be copied to the target network every _TARGET_UPDATE_FREQ_ epochs. The network updates are performed by _torch.autograd_, the built-in differentiation engine of PyTorch. The latest model will be saved every 10000 iterations.

If _arg_=='resume', training will be resumed from the last saved checkpoint. The greedy policy of the current online network will be used to fill the buffer up to _MIN_REPLAY_SIZE_. Training will continue after that.
