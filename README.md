# Brain Tumor Classification using DeIT Transformer

The model has been fine-tuned up to 32 layers along with a redesigned head to leverage the full features of the DeIT architecture. The repository also includes the implementation of various techniques to optimize the model's performance, including the AdamW optimizer, early stopping, ReduceLROnPlateau learning rate scheduler, batch normalization, dropout, and ReLU activation function.

## Repository Contents

- **Dataset**: The dataset used in this study, including training, validation, and test images.
- **Published Paper**: The full published paper detailing the methodology, experiments, results, and discussions on brain tumor classification using the DeIT model.
- **Code**: The complete codebase for training, validating, and testing the brain tumor classification model.

## Model Overview

The brain tumor classification model is built on the Data-efficient Image Transformer (DeIT), fine-tuned up to 32 layers with a redesigned head for improved classification accuracy. The model is designed to classify brain tumor images into multiple categories with high precision and recall.

### Key Features

- **Data-efficient Image Transformer (DeIT)**: Utilizes the full capabilities of the DeIT architecture, fine-tuned up to 32 layers.
- **Redesigned Head**: The head of the model is redesigned to enhance classification performance.
- **AdamW Optimizer**: Used for efficient optimization during training.
- **Early Stopping**: Prevents overfitting by stopping the training process when performance on the validation set ceases to improve.
- **ReduceLROnPlateau**: Adjusts the learning rate dynamically based on validation performance to ensure efficient convergence.
- **Batch Normalization**: Normalizes the activations of the neurons to improve the training process.
- **Dropout**: Regularization technique to prevent overfitting by randomly dropping units during training.
- **ReLU Activation Function**: Applied to introduce non-linearity in the model, helping to learn complex patterns.
- **Contour Analysis for Cropping**: Uses contour analysis and end point detection to crop the images, focusing on regions of interest and removing irrelevant background data.

## Usage

### Requirements

- Python 3.8+
- PyTorch
- Transformers Library
- NumPy
- Scikit-learn
- Other dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Shankar-Singh-Mahanty/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Model

1. **Data Preparation**: Ensure the dataset is properly structured in the dataset directory.

2. **Training**: Run the training script to start the model training process.

3. **Validation**: Validate the trained model using the validation script.

4. **Testing**: Test the model performance on the test dataset.

### Results

The model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. The results are documented in the published paper included in this repository.

## Contributing

Contributions to this repository are welcome. Please open an issue or submit a pull request for any improvements or suggestions.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

We acknowledge the use of the DeIT model and the various optimization techniques implemented in this study. Special thanks to the contributors of the Transformers Library and the PyTorch community for their valuable tools and resources.

---

For more detailed information, please refer to the published paper included in this repository. If you have any questions or need further assistance, feel free to open an issue or contact us directly.
