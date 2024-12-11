# Tokamak Fusion Reactor Optimization

**Final Project, Computational Physics, Fall 2024**

This project uses reinforcement learning to optimize the position and plasma current within a tokamak, inspired by and based upon the research and findings of the [TCV fusion experiments](https://github.com/MrLou1976/PHY_329_Final_Project/blob/main/Useful%20Information/Magnetic%20control%20of%20tokamak%20plasmas%20through%20deep%20reinforcement%20learning.pdf). 

Please refer to the **Useful Information** directory for resources. We are primarily referring to the [*Magnetic control of tokamak plasmas through deep reinforcement learning*](https://github.com/MrLou1976/PHY_329_Final_Project/blob/main/Useful%20Information/Magnetic%20control%20of%20tokamak%20plasmas%20through%20deep%20reinforcement%20learning.pdf) and [*Supplementary Information*](https://github.com/MrLou1976/PHY_329_Final_Project/blob/main/Useful%20Information/Supplementary%20Information.pdf) documents. Additionally, we will base our code on the files provided in the **fusion_tcv** directory.

# Introduction

The tokamak is one of the more popular and efficient structures used in modern nuclear fusion reactors. Our goal is to train a PyTorch RL model with the source data gathered in the TCV experiments, then use the FreeGS tokamak simulator to test and display our results.

In order to accomplish this goal, we need to define a reward function to use in our RL model to encourage the plasma to take on a more favorable configuration, such as those outlined by the TCV experiments. 

The original experiments were much more sophisticated than our implementation hopes to be, as we plan to primarily focus on optimizing plasma current, whereas the original experiments also gathered information regarding position, shape, and growth rate.

# Installation & Example Usage

First, you'll need to install all of the dependencies listed in [requirements.txt](https://github.com/MrLou1976/PHY_329_Final_Project/blob/main/requirements.txt):

*(Certain dependencies may interfere with each other. Use Python 3.10.11 to get around this if necessary)*

```
pip install -r requirements.txt
```



