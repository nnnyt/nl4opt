# NL4Opt Subtask 2

The source code of Team Long for the subtask 2 of NeurIPS 2022 Competition: The Natural Language for Optimization (NL4Opt). The final score is 0.867, which won the third place in the competition. Please refer to [NL4Opt](https://nl4opt.github.io) for more information of the competition.

Our code is based on the provided [baseline](https://github.com/nl4opt/nl4opt-subtask2-baseline).

## Environment
We have provided a Conda environment file environment.yml. To install:
```bash
conda env create -f environment.yml
conda activate tgen
```

Verify that it was installed:

```bash
conda env list
```

## Getting Started
* Download the dataset

  Dataset files can be found in [this dataset repository](https://github.com/nl4opt/nl4opt-competition/tree/main/generation_data). Copy the dataset files `train.jsonl` and `dev.jsonl` to the `data` subdirectory. Please change the config files if dataset files are in other paths.

* Training & Testing

  The training and testing pipeline can be run using `train_and_evaluate.sh`.

  ```bash
  ./train_and_evaluate.sh
  ```

