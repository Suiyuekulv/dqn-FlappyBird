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
This program has been tested in Anaconda virtual environment.
1. Open Anaconda Prompt 
 
3. Install Torch from [PyTorch](https://pytorch.org/):

CUDA 11.3
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
CPU
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```


