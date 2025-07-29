# 🐍 Snake Game AI - Reinforcement Learning with PyTorch

This project is a classic **Snake game** implemented using **Pygame**, trained with a **Deep Q-Network (DQN)** agent using **PyTorch**. The AI learns how to play Snake from scratch through self-play using reinforcement learning techniques.

---

## 🎮 Demo

![Game Preview](assests/Demo.png)

---

## 📁 Project Structure

snake-pygame/
│
├── agent.py # Core agent logic (DQN-based)
├── game.py # Snake game environment (Pygame-based)
├── model.py # Neural network model and trainer
├── helper.py # Real-time score plotting using matplotlib
├── model/ # Saved trained models (.pth files)
└── README.md # Project documentation

## 🧠 How It Works

The AI agent learns using the **Deep Q-Learning** algorithm:
- **State Space (11 features)**: Danger detection, current direction, and food location.
- **Action Space (3 actions)**: [Straight, Right, Left]
- **Reward Mechanism**:
  - `+10` for eating food
  - `-10` for dying
  - `0` for regular move
- **Experience Replay**: Uses a memory buffer to sample random experiences for stable training.
- **Neural Network**:
  - Input: 11 neurons
  - Hidden Layer: 200 neurons (ReLU)
  - Output: 3 neurons (Q-values for actions)

---

## 📈 Live Plotting

The training progress (score & mean score) is visualized in real-time using `matplotlib`.

![Game Preview](assests/Graph.png)

---

## 🧪 Requirements

- Python ≥ 3.7
- PyTorch
- Pygame
- Matplotlib
- NumPy
