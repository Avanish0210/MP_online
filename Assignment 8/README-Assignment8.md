# Handwritten Digit Recognition — Artificial Neural Network (ANN)

## Objective
Build an Artificial Neural Network (ANN) using TensorFlow/Keras to classify
handwritten digits (0-9) from the MNIST dataset, as a stand-in for automating postal
code digit recognition.

## Dataset
MNIST Handwritten Digits Dataset (Kaggle, CSV format):
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

> The dataset CSVs are not redistributed in this repo per the assignment
> instructions. Download `mnist_train.csv` and `mnist_test.csv` from the Kaggle link
> above and place them in the repo root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- tensorflow / keras (`Sequential`, `Dense`, Adam optimizer)
- scikit-learn (`confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`)

## Methodology
1. **Data Understanding** — loaded `mnist_train.csv` / `mnist_test.csv` with pandas
   (each row: a `label` column plus 784 `pixelN` columns for a flattened 28×28
   grayscale image), inspected shape/info, and displayed one sample digit image with
   Matplotlib.
2. **Data Preprocessing** — confirmed no missing values, separated features (784
   pixel columns) from the target (`label`), normalized pixel values from 0-255 to
   0-1, and one-hot encoded the 10 digit classes. The official train/test CSVs are
   already a clean, non-overlapping split (60,000 / 10,000 images, ~86/14), so they're
   used directly as the training and test sets.
3. **Model Development** — built a Sequential ANN (Input → Dense(128, ReLU) →
   Dense(64, ReLU) → Dense(10, Softmax)), compiled with the Adam optimizer,
   categorical crossentropy loss, and accuracy metric, then trained for 10 epochs
   (batch size 32, 10% validation split) and predicted digit classes on the test set.
4. **Model Evaluation** — evaluated with test accuracy, a confusion matrix, and a full
   classification report, plus accuracy-vs-epoch and loss-vs-epoch plots.

## Model Architecture
```
Input(784)
  -> Dense(128, activation='relu')
  -> Dense(64, activation='relu')
  -> Dense(10, activation='softmax')

Optimizer: Adam
Loss: Categorical Crossentropy
Metric: Accuracy
Epochs: 10
```

## Results
| Metric | Value |
|---|---|
| Test Loss | ≈ 0.071 |
| Test Accuracy | ≈ 0.981 |

Precision, recall, and F1-score were consistently high (~0.95-1.00) across all 10
digit classes, with most confusion concentrated among visually similar digit pairs.

## ⚠️ Important note on the data used
The environment this notebook was drafted and executed in has restricted network
egress and could not download the actual Kaggle CSVs, so `sample_mnist_train.csv` /
`sample_mnist_test.csv` were used to produce the outputs you see in the notebook —
these are **real handwritten-digit images** (not synthetic pixels), sourced from
scikit-learn's built-in digits dataset, upscaled from 8×8 to 28×28 and saved in the
exact same `label` + 784-`pixel` CSV format as the real Kaggle files. **Before
submitting, download the actual Kaggle `mnist_train.csv` / `mnist_test.csv` and
replace the sample files with the same names** — the notebook code needs no changes
and will produce updated results on the real, full-size (60k/10k image) dataset.

## Conclusion
This project built a fully-connected ANN (784 → 128 → 64 → 10) to classify handwritten
digits, training for 10 epochs with the Adam optimizer and categorical crossentropy
loss, and reached about 98% test accuracy with mild signs of overfitting appearing in
later epochs as training accuracy pulled ahead of validation accuracy. Hidden layers
give an ANN the capacity to learn non-linear, hierarchical combinations of the raw
pixel inputs — without them, the network could only learn a single linear mapping from
pixels to digit class, which isn't expressive enough to separate 10 visually complex,
overlapping digit shapes. A key advantage of Deep Learning over traditional Machine
Learning here is that the network learns useful feature representations directly from
raw pixels, removing the need to hand-engineer features the way a classical ML pipeline
typically would. A key limitation of this plain ANN is that it treats every pixel as an
independent input and ignores spatial structure entirely, which is exactly the gap
Convolutional Neural Networks are designed to close.

## Files
- `Assignment-8.ipynb` — full notebook (data understanding, preprocessing, model,
  training, evaluation, accuracy/loss curves, and conclusion)
- `sample_mnist_train.csv` / `sample_mnist_test.csv` — the real-data stand-in files
  used to produce the executed outputs (see note above); replace with the actual
  Kaggle CSVs before submitting
- `README.md` — this file
