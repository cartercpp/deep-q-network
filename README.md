# Deep Q-Network from Scratch in C++

A from-scratch **Deep Q-Network (DQN)** that learns to walk a maze in modern C++.

Instead of a tabular Q-table, the agent uses a neural network to approximate Q-values for each action. No machine learning libraries are used.

This is part of the [ml-from-scratch](https://github.com/cartercpp/ml-from-scratch) collection.

## How It Works

The maze is a hardcoded `10x15` grid. The agent starts in the top-left cell and needs to reach the bottom-right goal.

Each state is encoded as a 6-dimensional vector:

```text
[normalized row, normalized column, can go up, can go down, can go left, can go right]
```

The network maps that encoding to four Q-values:

```text
6 inputs
    ↓
32 ReLU
    ↓
32 ReLU
    ↓
32 ReLU
    ↓
4 Q-values  (UP, DOWN, LEFT, RIGHT)
```

Training follows the classic DQN pattern:

* an **online** network chooses actions and is updated every step
* a **target** network provides stable future Q-values
* experiences `(state, action, reward, next state, done)` go into a replay buffer
* once the buffer has 500 samples, each step trains on 32 random experiences
* the target network is copied from the online network every 500 frames

The Bellman target used for the taken action is:

```text
Q(s, a) = r                     if the episode is done
Q(s, a) = r + γ * max Q_target(s', *)   otherwise
```

with `γ = 0.99`.

## Rewards and Exploration

| Event | Reward |
|---|---|
| Ordinary step | `-0.01` |
| Walk into a wall or off the map | `-1` (agent stays put) |
| Reach the goal | `+1` |

Action selection is epsilon-greedy. Epsilon starts at `1.0`, multiplies by `0.9999` after each step, and floors at `0.05`.

## Training

The `training/` directory contains the live training loop.

1. Encode the current cell.
2. Choose an action with epsilon-greedy policy on the online network.
3. Step the maze and store the experience.
4. Sample a minibatch from the replay buffer.
5. Compute the DQN target with the target network.
6. Call `onlineNN.fit(...)` so the taken action's Q-value moves toward that target.
7. Periodically copy the online weights into the target network.
8. Press **Enter** to stop. Weights and biases are written to `weights.txt` and `biases.txt`.

Network size and learning rate:

```cpp
neural_network onlineNN({6, 32, 32, 32, 4}, 0.001);
```

Replay buffer cap: `10,000` experiences.

## Demo

The `demo/` directory loads the trained weights and plays greedy policy in the terminal.

You will see:

- yellow `_` — start
- green `_` — goal
- red `█` — wall
- cyan `o` — the agent

The agent repeatedly walks from start to goal. Press **Enter** to quit.

## Project Structure

```text
deep-q-network/
├── training/
│   ├── main.cpp
│   ├── neural_network.h
│   ├── neural_network.cpp
│   ├── matrix.h
│   ├── math_vector.h
│   ├── weights.txt
│   └── biases.txt
│
└── demo/
    ├── main.cpp
    ├── mazeNN.cpp
    ├── neural_network.h
    ├── neural_network.cpp
    ├── matrix.h
    ├── math_vector.h
    ├── weights.txt
    └── biases.txt
```

## Implementation

The project implements the linear algebra and the network itself:

* `matrix.h` / `math_vector.h` — vectors, matrices, and the ops needed for backprop
* `neural_network` — multilayer perceptron with ReLU hidden layers, `predict` and `fit`
* `training/main.cpp` — maze, encoding, replay buffer, target network, DQN update

No TensorFlow, PyTorch, Eigen, or other ML libraries.

## Requirements

A modern C++ compiler with C++20/C++23 support (`std::jthread`, `<format>`, `std::to_underlying`).

A terminal that understands ANSI / truecolor helps for the demo.

## Why?

Tabular Q-learning needs one row per cell. A DQN approximates the same function with a small network, so the agent can generalize from a compact state encoding instead of memorizing every square.
