# Building and Deploying Question Answering System with Hugging Face
For better view of output, kindly run (Karthik_AIML_M18.ipynb) file in Google Colab
---
In this project, building and deploying a question-answering system using Hugging Face's powerful libraries and tools was attempted. The focus is on leveraging pre-trained models, specifically DistilBERT, fine-tuning it on the SQuAD (Stanford Question Answering Dataset), and deploying the trained model using Gradio for a user-friendly interface. The process involves several steps: installing necessary packages, loading datasets, preprocessing data, training the model, evaluating the model, and deploying the model.

## Installation of Required Libraries

- **transformers, datasets, evaluate**: For accessing and evaluating datasets and models.
- **transformers[torch]**: For deep learning tasks with PyTorch.
- **accelerate**: To enable faster training on multiple GPUs.
- **gradio**: For creating web interfaces for ML models.

## Hugging Face Model Hub Interaction

- **notebook_login**: Log in to Hugging Face for accessing models and datasets.

## Dataset Loading and Splitting

- **load_dataset**: Load the SQuAD dataset and split it into training and testing subsets.

## Tokenization and Preprocessing

- **AutoTokenizer**: Load the DistilBERT tokenizer.
- **preprocess_function**: Preprocess dataset examples, including truncation, padding, and mapping start and end tokens.

## Dataset Mapping and Batching

- **map**: Apply preprocessing over the entire dataset.
- **DefaultDataCollator**: Create batches of examples for training.

## Model Initialization and Training

- **AutoModelForQuestionAnswering**: Load the DistilBERT model for question answering.
- **TrainingArguments**: Define training parameters such as learning rate, batch size, number of epochs, and weight decay.
- **Trainer**: Train the model using the specified arguments and datasets.

## Model Training and Evaluation

- **trainer.train**: Train the model.
- **trainer.push_to_hub**: Share the trained model on Hugging Face Model Hub.

## Model Inference and Evaluation

- **pipeline**: Create a question-answering pipeline with the trained model.
- Evaluate the model on a subset of the SQuAD validation dataset.
- Compute metrics such as Exact Match (EM) and F1 Score.

## Web Interface Creation with Gradio

- **gr.Interface**: Create an interactive web interface for the question-answering model.
- **iface.launch**: Launch the Gradio interface for user interaction.

This system demonstrates the end-to-end process of building, training, and deploying a question-answering model using Hugging Face libraries and tools. It includes dataset preparation, model fine-tuning, evaluation, and creating a web interface for user interaction.
