
# AiDTA

![Figure 3](figure3-ASSEMBLE.png)

## Overview
This project implements a policy-value network and a self-play pipeline for generating and training data for decision-making in a sequence-based task. The main components include a neural network architecture, a data collection pipeline, a Monte Carlo Tree Search (MCTS) implementation, and configuration settings.

## File Descriptions

### `cnn_net.py`
This file defines the core neural network architecture, including the policy-value network and its training logic. Key components include:
- **ResBlock**: Implements a residual block for feature extraction.
- **Net**: Defines the policy-value network with a shared feature extractor and separate policy and value heads.
- **PolicyValueNet**: A wrapper class for training and inference with the `Net` model. Includes methods for:
  - Forward propagation to output action probabilities and state values.
  - Training with data from Monte Carlo Tree Search (MCTS).
  - Saving and loading the trained model.

### `collect.py`
This file contains the logic for collecting self-play data to train the policy-value network. Key components include:
- **CollectPipeline**: Manages the self-play process, including:
  - Loading the model.
  - Running simulations using MCTS.
  - Storing collected data in a replay buffer for training.
- **Self-play Data Collection**: Uses legal actions generated by the game logic to simulate games and gather data for training.
- **Execution**: The `run()` method continuously collects data until interrupted.

### `config.py`
Contains configuration settings for the project, including:
- Neural network and training parameters (e.g., learning rate, batch size, KL divergence target).
- Paths for saving and loading models and data buffers.
- Parameters for MCTS simulations (e.g., Dirichlet noise, exploration weights).

### `mcts.py`
This file implements the Monte Carlo Tree Search (MCTS) algorithm used for decision-making. Key components include:
- **TreeNode**: Represents a node in the MCTS tree. Includes methods for:
  - Expanding the tree with child nodes.
  - Selecting the optimal child node based on Q-value and exploration.
  - Updating node values recursively based on leaf evaluations.
- **MCTS**: The core MCTS implementation. Includes:
  - Performing simulations (playouts) to explore the search space.
  - Evaluating game states using the policy-value network.
  - Computing action probabilities based on visit counts.
- **MCTSPlayer**: An AI player that uses MCTS for making decisions. Includes:
  - Resetting the search tree.
  - Generating actions based on MCTS results with exploration noise.
  - Updating the tree with chosen moves.

### `game.py`
This file provides the core logic for sequence manipulation, state representation, and game evaluation. Key components include:
- **availabel**: Generates all possible insertion sequences and structures at valid positions in the current sequence.
  - **Inputs**: Current sequence, structure, and fragment lists (`list1` and `list2`).
  - **Outputs**: Moves, updated sequences, and structures.
- **State**: Encodes a given sequence and structure into a feature matrix suitable for neural network input.
  - Combines nucleotide and secondary structure information into a 6x61 matrix.
- **game_end**: Evaluates the similarity between the predicted secondary structure and the target structure using RNAstructure tools.
  - Returns a score of 1 if similarity exceeds 90%; otherwise, returns -1.
- **self_play**: Simulates a complete self-play game, recording sequences, structures, MCTS probabilities, and final scores.
  - Terminates when the sequence reaches a predefined length or achieves a high similarity score.
  - Stores the best sequences and structures in `best_sequence.txt`.

## How to Use

### Prerequisites
- Python 3.7 or higher
- Required libraries: `torch`, `numpy`, `pickle`

### Setup
1. Install the required Python libraries:
   ```bash
   pip install torch numpy
   ```
2. Ensure the necessary files (`cnn_net.py`, `collect.py`, `config.py`, `mcts.py`, `game.py`) are in the same directory.
3. Update the paths in `config.py` if necessary (e.g., model and data buffer paths).

### Running the Project

#### Collect eligible sequences and train model
1. Ensure the `CollectPipeline` class in `collect.py` is correctly initialized with a valid model path (`current_policy.pkl`) or starts with a new model.
2. Run the script to begin the self-play and data collection pipeline:
   ```bash
   python collect.py
   ```
3. The pipeline will continuously collect self-play data and store it in a buffer for training.
4. Run the training script:
   ```bash
   python train.py
   ```

#### After Collecting and Filtering Fragments
Once the sequence fragments and their secondary structures have been filtered, the filtered fragments and their corresponding secondary structures should be placed into `list1` and `list2` in the respective files (`game.py`, `mcts.py`, and `collect.py`). Here's an example of how these lists should look:

```python
# Fragment list
list1 = [
    'GGC&GCC', 'CGT&ACG', 'TAT&ATA', 'CAC&GTG', 'AAG&CTT', 'TACA&TGTA', 
    'ATCG&CGAT', 'AATA&TATT', 'TAAA&TTTA', 'GTGG&CCAC', 'GGGTG&CACCC', 
    'CCAGC&GCTGG', 'CGGTG&CACCG', 'AGGTG&CACCT', 'TCAGG&CCTGA', 
    'TAAATG&CATTTA', 'AAAAGG&CCTTTT', 'TAAAGG&CCTTTA', 'TAAGTG&CACTTA', 
    'TAAAGT&ACTTTA', 'AGG', 'CTG', 'GGG', 'GAA', 'GAC', 'GAGA', 'GCCA', 
    'ATTT', 'TCTG', 'GTTG', 'TTAGT', 'TTAGA', 'TTGGT', 'TTGGA', 'TTTGC', 
    'TTTTAA', 'TCTTTG', 'TTTTAC', 'TACGTC', 'TTCTGG'
]

# Structure of fragment
list2 = [
    '(((&)))', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '((((&))))', 
    '((((&))))', '((((&))))', '((((&))))', '((((&))))', '(((((&)))))', 
    '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '((((((&))))))', 
    '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', 
    '...', '...', '...', '...', '...', '....', '....', '....', '....', 
    '....', '.....', '.....', '.....', '.....', '.....', '......', '......', 
    '......', '......', '......'
]
```

These two lists (`list1` and `list2`) will be used across the relevant scripts to perform the necessary operations.

### Inference
Use the `PolicyValueNet` class in `cnn_net.py` for making predictions on sequences and structures. For example:
```python
from cnn_net import PolicyValueNet

model = PolicyValueNet(model_file='current_policy.pkl')
sequence = 'AAACCCAAACCCCCCAAAAAA&GGG&GGGAAACCCCCC&GGG&GGG&GGGAAA'
structure = '...(((...((((((......&)))&)))...((((((&)))&)))&)))...'
act_probs, value = model.policy_value([sequence, structure])
print("Action Probabilities:", act_probs)
print("Value:", value)
```

## Project Structure
- **Neural Network**: Implements a residual CNN architecture for policy and value prediction.
- **Self-play Pipeline**: Simulates games using MCTS and collects data for training.
- **Monte Carlo Tree Search**: Guides decision-making with simulations and exploration.
- **Game Logic**: Manages sequence generation, state encoding, and evaluation.
- **Configuration**: Centralized settings for reproducibility and customization.

## Notes
- The `game` module, which contains key logic for the sequence and structure generation, is referenced but not provided in this repository.
- To ensure compatibility, the `CONFIG` dictionary should align with the specifics of your task (e.g., number of channels, sequence lengths).

## Future Work
- Optimize the pipeline for larger datasets and longer training durations.
- Extend the `game` module with additional logic for more complex tasks.
- Incorporate advanced techniques like distributed training for scalability.

## Contact
For questions or issues, please reach out to the project maintainer.

---

**Author**: Guo Gaoxing  
**Date**: January 19, 2025
```

---

