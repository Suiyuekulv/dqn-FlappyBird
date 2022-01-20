# Playing FlappyBird with double deep Q learning

Play the game 'Falppy Bird' using double deep Q learning. The deep neural networks are implemented by PyTorch.

## Introduction
This prject creates an intelligent agent that automatically learns how to play the game 'Flappy Bird', by following a double deep Q-learning scheme, which is described in _Deep Reinforcement Learning with Double Q-learning_[1]. By introducing the technique of experience replay, the algorithm performs well in learning stability. 

## Dependencies
- Python 3.9
- PyTorch 1.10.1
- pygame
- OpenCV

## Setup
_This program is to be run in Anaconda virtual environment._ Open Anaconda Prompt and go to a prefered directory for the following setup:
 
1. Install Torch from [PyTorch](https://pytorch.org/):

 CUDA 11.3
 ```
 conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
 ```
 CPU
 ```
 conda install pytorch torchvision torchaudio cpuonly -c pytorch
 ```

2. Install pygame
 ```
 python3 -m pip install -U pygame --user
 ```

3. Install OpenCV
 ```
 conda install -c conda-forge opencv
 ```
