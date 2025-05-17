# Parallel Best-First Minimax Search

This project implements a **parallelized version of the Best-First Minimax Search algorithm**, designed to efficiently evaluate game trees in two-player, zero-sum games. By leveraging multithreading, the algorithm aims to improve performance over traditional sequential approaches.

## 📌 Overview

The Best-First Minimax Search explores the most promising nodes first, often reducing the number of evaluations needed compared to depth-first search. This implementation parallelizes the search to further speed up decision-making in game environments.

## 🚀 Features

- 🔄 **Parallel Processing** using multithreading (e.g. `std::thread`)
- 🧠 **Heuristic Evaluation** of non-terminal nodes
- 🧩 **Modular Design** to allow easy experimentation and integration

## 🛠️ Getting Started

### Prerequisites

- C++11 or later
- CMake 3.10+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abdulrahman-elnabawi/Parallel-Best-First-Minimax-Search.git
