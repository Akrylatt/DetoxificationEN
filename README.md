# Text Detoxification Repository

This repository contains the solution for the Text Detoxification Task, which is the process of transforming toxic-style text into neutral-style text while preserving the same meaning.

**Author:** Adela Krylova
**Email:** a.krylova@innopolis.university
**Group Number:** BS20-AI-01

## About the Text Detoxification Task

The Text Detoxification Task involves creating a solution for detoxing text with a high level of toxicity. This can be achieved through various models, algorithms, or techniques. The data for this task is labeled with binary classification (toxic/non-toxic) by annotators, resulting in a toxicity dataset. The dataset also includes pairs of text, where one has a high toxicity level and the other is a paraphrased version with low toxicity. This pair structure aids in training models to distinguish between the overall meaning of the text and its toxicity level.

For more details about the task, you can refer to the formal definition in "Text Detoxification using Large Pre-trained Neural Models" by Dale et al., page 14.

## Repository Structure

The repository is organized as follows:

- `data`: Contains directories for external data, interim data, and raw data.
- `models`: Holds trained and serialized models, as well as final checkpoints.
- `notebooks`: Includes Jupyter notebooks for data exploration and the final solution.
- `references`: Contains data dictionaries, manuals, and explanatory materials.
- `reports`: Contains generated analysis in HTML, PDF, LaTeX, etc., along with figures used in reporting.
- `requirements.txt`: Lists the requirements for reproducing the analysis environment.
- `src`: Contains the source code for data processing, model training, and visualization.

## How to Use This Repository

### 1. Data Processing

To transform the data, you can use the provided scripts in the `src/data` directory. For example, you can run `make_dataset.py` to generate the necessary data for your model.

### 2. Model Training

In the `src/models` directory, you'll find scripts for training and making predictions using trained models. You can use `train_model.py` to train your models and `predict_model.py` for making predictions.

### 3. Visualization

For exploratory and results-oriented visualizations, you can use the script in the `src/visualization` directory. It's a good practice to visualize your data and model results to gain insights.

## Reports

This repository includes two reports: 

1. The first report describes the path in the solution creation process.

2. The second report contains details about the final solution.

## Notebooks

In the `notebooks` directory, you will find four notebooks:

1. `1.0-initial-data-exploration.ipynb`: This notebook contains your initial data exploration.

2. `2_0_dataset_preparation.ipynb`: This notebook contains information about basic ideas behind data preprocessing.
  
3. `3_0_model_fine_tuning.ipynb`: This notebook contains the details of the model fine-tuning

4. `4_0_testing_answers.ipynb`: This notebook contains visualisation of the final translation and some examples


## Questions and Feedback

If you have any questions, feedback, or suggestions, please feel free to reach out to the author at a.krylova@innopolis.university.

Thank you for your interest in this Text Detoxification project!

