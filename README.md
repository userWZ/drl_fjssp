# Optimizing Fuzzy Job Shop Scheduling Using Graph Neural Networks and Deep Reinforcement Learning

This repository is the official PyTorch implementation of the paper "Optimizing Fuzzy Job Shop Scheduling Using Graph Neural Networks and Deep Reinforcement Learning" 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements the method proposed in the paper "Optimizing Fuzzy Job Shop Scheduling Using Graph Neural Networks and Deep Reinforcement Learning". The method demonstrates good performance in scheduling efficiency compared to all Priority Dispatching Rules (PDRs), with commendable solving time and good generalization across various instance scales in FJSSP.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the code, follow these steps:

1. **Set the experiment scale and generate random evaluation instances:**

    Configure the parameters in `params.py` to set the scale of your experiment and generate random evaluation instances.

2. **Use a pre-trained model or retrain the scheduling model:**

    - To use a pre-trained model, ensure the model files are placed in the appropriate directory specified in `params.py`.
    - To retrain the model, follow the instructions in the [Experiments](#experiments) section.

3. **Execute the solving process:**

    Once the model is ready, execute the main solving script:

    ```bash
    python eval.py 
    ```


## Experiments

To reproduce the experiments from the paper, follow these steps:

1. Ensure your environment is set up as described in the [Installation](#installation) section.
2. Configure the parameters in `params.py`.
3. Run the experiment scripts by:
    ```bash
    python train_ppo.py --config configs/train_config.yaml
    ```
    or:
    ```bash
    bash scripts/train.bash
    ```

   

## Results

The results of the experiments can be found in the `results` directory. Each experiment script will generate output files containing the performance metrics and visualizations.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


