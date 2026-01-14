# IITMDSA_Final_Project

# 1st. Sequence-to-Sequence-Modeling-with-Attention-Mechanism
A Sequence-to-Sequence (Seq2Seq) model with attention mechanism for sequence transformation tasks. The project includes data generation, training, and evaluation scripts, utilizing PyTorch for implementation. Features modular code structure with CSV-based dataset handling and visualization of training loss curves.

---

# Seq2Seq Model with Attention Mechanism

This project implements a Sequence-to-Sequence (Seq2Seq) model with an attention mechanism for sequence transformation tasks using PyTorch. The model is trained on synthetic data and evaluates performance in reversing sequences. 

## Features
- **Seq2Seq Architecture**: Includes Encoder, Attention, and Decoder components.  
- **Attention Mechanism**: Enables better handling of long input sequences by focusing on relevant parts.  
- **Synthetic Dataset**: Automatically generates source and target sequences for training.  
- **Training and Testing**: Includes scripts for training the model and evaluating accuracy.  
- **Visualization**: Plots the training loss curve for monitoring progress.  
- **Modular Design**: Organized with separate files and folders for models, utilities, and data.  

## Project Structure
```
Seq2Seq_Model_Project/
│
├── data/                     # Dataset-related files
│   └── dataset.csv           # Source and target sequences
│
├── models/                   # Model definitions
│   ├── encoder.py            # Encoder class
│   ├── attention.py          # Attention class
│   ├── decoder.py            # Decoder class
│   └── seq2seq.py            # Seq2Seq model
│
├── utils/                    # Utility functions
│   └── data_processing.py    # Data generation and processing
│
├── train.py                  # Training script
├── test.py                   # Testing script
├── main.py                   # Script to execute the full workflow
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── notebooks/                # (Optional) Experimentation notebooks
    └── seq2seq_model.ipynb
```

## Results
### Training Loss
The training loss decreased over 10 epochs, showing effective learning:
- **Epoch 1 Loss**:  2.16299
- **Epoch 5 Loss**:  1.17597
- **Epoch 10 Loss**: 0.26381  

### Test Accuracy
The model achieved **94.33% accuracy** on the test dataset.

## Prerequisites
- Python 3.9 or higher
- CUDA-enabled GPU (optional for faster training)

## Installation
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd Seq2Seq_Model_Project
   ```
2. Set up a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Generate Dataset**:  
   Run the data processing script to create `dataset.csv` in the `data/` folder:  
   ```bash
   python utils/data_processing.py
   ```
2. **Train the Model**:  
   Train the Seq2Seq model using `train.py`:  
   ```bash
   python train.py
   ```
3. **Test the Model**:  
   Evaluate accuracy on the test dataset:  
   ```bash
   python test.py
4. **Get the results**:  
   Get accuracy and loss curve on the test dataset:  
   ```bash
   python main.py
   ```

## Dependencies
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

For more details, see `requirements.txt`.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2nd. Comparison of CNN Architectures on Different Datasets




## Skills Takeaway From This Project


In this project, learners will gain skills in:
- Understanding and implementing various Convolutional Neural Network (CNN) architectures
- Applying CNNs to different types of datasets
- Evaluating model performance using various metrics
- Analyzing the impact of dataset characteristics on model performance
- Utilizing popular deep learning frameworks such as PyTorch and TensorFlow



    
## 6Domain

Machine Learning, Deep Learning, Computer Vision




## Problem Statement


The goal of this project is to compare the performance of different CNN architectures on
various datasets. Specifically, we will evaluate LeNet-5, AlexNet, GoogLeNet, VGGNet,
ResNet, Xception, and SENet on MNIST, FMNIST, and CIFAR-10 datasets. The comparison
will be based on metrics such as loss curves, accuracy, precision, recall, and F1-score.



    
## Business Use Cases


The insights from this project can be applied in various business scenarios, including:
- Choosing the appropriate CNN architecture for specific computer vision tasks
- Improving model performance by understanding the impact of dataset characteristics
- Optimizing resource allocation by selecting models that offer the best trade-off between
performance and computational cost



    
## Approach


1. Load and preprocess the datasets (MNIST, FMNIST, CIFAR-10).
2. Implement the following CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet,
ResNet, Xception, and SENet.
3. Train each model on each dataset, recording the loss and accuracy metrics.
4. Evaluate the performance of each model on the test sets using accuracy, precision, recall,
and F1-score.
5. Plot the loss curves and other performance metrics for comparison.
6. Analyze the results to understand the impact of different architectures and datasets on
model performance.



    
## Results


The expected outcomes of this project include:
- Comparative loss curves for each model on each dataset
- Accuracy, precision, recall, and F1-score for each model on each dataset
- Analysis of the results to determine the strengths and weaknesses of each architecture on
different datasets

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3rd. Multifunctional-NLP-and-Image-Generation-Tool-using-Hugging-Face-Models


This project is a Streamlit-based web application that provides multiple functionalities using Natural Language Processing (NLP) and Image Generation models. Users can choose from various tasks such as text summarization, next word prediction, story generation, chatbot interactions, sentiment analysis, question answering, and image generation.


# Features


Text Summarization: Summarize large bodies of text into concise summaries.

Next Word Prediction: Predict the continuation of a sentence.

Story Prediction: Generate creative stories based on given prompts.

Chatbot: Engage in conversation with an AI-powered chatbot.

Sentiment Analysis: Analyze the sentiment (positive/negative) of a given text.

Question Answering: Provide answers to questions based on provided context.

Image Generation: Create images based on textual descriptions.


# Models Used


Text Summarization: facebook/bart-large-cnn from Hugging Face.

Next Word Prediction & Story Prediction: gpt-neo model from Hugging Face.

Sentiment Analysis: Pretrained sentiment analysis pipeline from Hugging Face.

Question Answering: Pretrained question-answering pipeline from Hugging Face.

Image Generation: Stable Diffusion model from Hugging Face’s CompVis/stable-diffusion-v1-4.


# Usage


Select a Task: Choose the desired task from the sidebar.

Provide Input: Depending on the selected task, input the required text or prompt.

Execute Task: Click the corresponding button to get the output.

View Output: The output will be displayed below the input section.


## Project Structure
```
multifunctional-nlp-image-generation-tool/
├── app.py                   # Main Streamlit app with task integration
├── requirements.txt         # List of required dependencies
├── tasks/                   # Folder containing task-specific logic
│   ├── __init__.py
│   ├── text_summarization.py
│   ├── next_word_prediction.py
│   ├── story_prediction.py
│   ├── chatbot.py
│   ├── sentiment_analysis.py
│   ├── question_answering.py
│   └── image_generation.py
├── models/                  # Directory for model initialization
│   ├── __init__.py
│   ├── huggingface_loader.py
│   └── tokenizer.py
├── config/                  # Directory for configuration files
│   ├── __init__.py
│   ├── settings.py
│   └── constants.py
├── logs/                    # Logs for debugging and tracking
│   └── app.log
├── tests/                   # Test folder with task-specific unit tests
│   ├── __init__.py
│   ├── test_chatbot.py
│   ├── test_image_generation.py
│   ├── test_next_word_prediction.py
│   ├── test_question_answering.py
│   ├── test_sentiment_analysis.py
│   ├── test_story_prediction.py
│   ├── test_text_summarization.py
│   └── task_results.json      # Stores task evaluation results
└── README.md                # Project documentation
```

## Evaluation Metrics
### Task Performance

| Task                  | Accuracy | Precision | Recall | F1-Score | ROUGE-1 Recall | ROUGE-1 Precision | ROUGE-1 F1 | ROUGE-2 Recall | ROUGE-2 Precision | ROUGE-2 F1 | ROUGE-L Recall | ROUGE-L Precision | ROUGE-L F1 | Top-K Accuracy | BLEU Score | Exact Match |
|-----------------------|----------|-----------|--------|----------|----------------|-------------------|------------|----------------|-------------------|------------|----------------|-------------------|------------|-----------------|------------|-------------|
| Next Word Prediction  | 33.33%   | 33.33%    | 33.33% | 33.33%   | -              | -                 | -          | 66.67%         | -                 | -          | -              | -                 | -          | -               | -          | -           |
| Text Summarization    | -        | -         | -      | -        | 54.23%         | 24.29%            | 33.30%     | 16.04%         | 6.07%             | 8.73%      | 48.16%         | 21.46%            | 29.46%     | -               | -          | -           |
| Story Prediction      | 0.00%    | 16.67%    | 1.72%  | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | -          | -           |
| Chatbot               | -        | -         | -      | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | 2.16e-155  | -           |
| Sentiment Analysis    | 100%   | 100%    | 100% | 100%   | -              | -                 | -          |               | -                 | -          | -              | -                 | -          | -               | -          | 100%         |
| Question Answering    | -        | -         | -      | -        | -              | -                 | -          | -              | -                 | -          | -              | -                 | -          | -               | -          | 100%        |

Performance is validated through manual testing and feedback collection.

## Testing
### Manual Testing
Tested each task with valid and invalid inputs to verify:
- Accuracy of results.
- Robustness of error handling.

### Automated Testing
Incorporated unit tests for key functionalities in the `tests/` folder.

### Sample Test Cases
- **Text Summarization**: Provide a passage and validate the summary.
- **Sentiment Analysis**: Test various sentiments for correctness.
- **Question Answering**: Ask context-based questions to verify answers.
- **Image Generation**: Generate images based on descriptive prompts.

## Future Improvements
1. **Cloud Deployment**: Host the app on platforms like AWS or Azure.
2. **Model Fine-Tuning**: Adapt models to domain-specific datasets.
3. **Task Expansion**: Add new NLP tasks like translation and paraphrasing.
4. **Optimization**: Reduce response times for computationally intensive tasks.
5. **Enhanced UI**: Add real-time previews and performance metrics displays.

## Contributing
Contributions are welcome! If you wish to contribute:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a clear description of your changes.

