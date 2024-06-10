# Building and Deploying Question Answering System with Hugging Face
For better view of output, kindly run (Karthik_AIML_M18.ipynb) file in Google Colab
---
base_model: distilbert-base-uncased

Fine tuned model name: distilbertfinetuneHS5E8BHLR

---


# distilbertfinetuneHS5E8BHLR

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on SQuAD dataset.


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 3.0251        | 1.0   | 500  | 1.7268          |
| 1.4512        | 2.0   | 1000 | 1.4143          |
| 0.9326        | 3.0   | 1500 | 1.4345          |
| 0.6653        | 4.0   | 2000 | 1.5804          |
| 0.5143        | 5.0   | 2500 | 1.6401          |


### Framework versions

- Transformers 4.38.2
- Datasets 2.18.0
- Tokenizers 0.15.2
