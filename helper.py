import math
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, \
    f1_score
from sklearn.preprocessing import StandardScaler

path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset_split.xlsx"
xls = pd.ExcelFile(path)

lipidomics = pd.read_excel(xls, "Lipidomics")
metabolomics = pd.read_excel(xls, "Pareto Metabolomics")
proteomics = pd.read_excel(xls, "Pareto Proteomics")


def plot_confusion_matrix(y_test_, y_prediction_):
    """
    Plot the confusion matrix of the model
    :param y_test_: True labels
    :param y_prediction_: Predicted labels
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_test_, y_prediction_)
    # Plot the confusion matrix
    if len(np.unique(y_prediction_)) == 2:
        labels = ["Stable", "Decliner"]
    else:
        labels = ["Stable", "Slow Decliner", "Fast Decliner"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="GnBu")
    plt.show()


def get_omics_type(feature_name):

    if feature_name in lipidomics.columns:
        return "Lipidomics"
    elif feature_name in metabolomics.columns:
        return "Metabolomics"
    elif feature_name in proteomics.columns:
        return "Proteomics"
    return None

def standardize(X):
    """
    Standardize the data using the StandardScaler
    This is crucial for regularized models
    (otherwise features with larger scales dominate the regularization)
    :param X: Data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def evaluate_model(y_pred, y_true):
    """
    Evaluate the model
    :param y_pred: Predicted labels
    :param y_true: True labels
    """

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy of the model: {accuracy:.4f}')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f'Recall of the model: {recall:.4f}')
    precision = precision_score(y_true, y_pred, average='weighted')
    print(f'Precision of the model: {precision:.4f}')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 score of the model: {f1:.4f}')

    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred)


def get_train_test_split_RIDs():
    """
    Get the RIDs of the training and testing set from the file
    :return: the RIDs of the training and testing set
    """
    path = "C:/Users/mjgoj/Desktop/THESIS/data/RIDs.txt"
    with open(path, "r") as file:
        lines = file.readlines()
        training_RIDs = lines[1].split(",")
        testing_RIDs = lines[3].split(",")
        # Convert to integers
        training_RIDs = [int(x) for x in training_RIDs]
        testing_RIDs = [int(x) for x in testing_RIDs]
        return training_RIDs, testing_RIDs


def get_train_test_split_dataset(dataset):
    """
    Get the training and testing set from the dataset
    :param dataset: the dataset with the RID column present
    :return: the training and testing set
    """
    training_RIDs, testing_RIDs = get_train_test_split_RIDs()
    training_set = dataset[dataset["RID"].isin(training_RIDs)]
    testing_set = dataset[dataset["RID"].isin(testing_RIDs)]
    print("Training set shape: ", training_set.shape)
    print("Testing set shape: ", testing_set.shape)
    return training_set, testing_set


def resampling(x, y):
    """
    Resample the dataset to balance the classes

    Initial dataset is very imbalanced, with the majority class being the "Stable" class
    Therefore the tree always predicted the majority class
    """
    smote = SMOTE()
    x_resampled, y_resampled = smote.fit_resample(x, y)
    print("Resampled set shape: ")
    for dataset in [x_resampled, y_resampled]:
        print(dataset.shape)
    print("Resampled set class distribution: \n", y_resampled.value_counts())
    return x_resampled, y_resampled


def pareto_scaling(dataset_):
    """
    https://www.rdocumentation.org/packages/MetabolAnalyze/versions/1.3.1/topics/scaling
    https://uab.edu/proteomics/metabolomics/workshop/2014/statistical%20analysis.pdf
    Pareto scaling is often used in metabolomics.
    It scales data by dividing each variable by the square root of the standard deviation,
    so that each variable has variance equal to 1.
    :param dataset_: original non-normalized dataset (can have RID and class labels)
    :return: normalized dataset
    """
    # Maintain a list of dropped columns to add them back after scaling
    dropped_columns = []
    if "RID" in dataset_.columns:
        RID = dataset_["RID"]
        dataset_ = dataset_.drop("RID", axis=1)
        dropped_columns.append(RID)
    if "ThreeClass" in dataset_.columns:
        three_class = dataset_["ThreeClass"]
        dataset_ = dataset_.drop("ThreeClass", axis=1)
        dropped_columns.append(three_class)
    if "TwoClass" in dataset_.columns:
        dataset_ = dataset_.drop("TwoClass", axis=1)

    # Scale the dataset
    for feature in dataset_.columns:
        std = dataset_[feature].std()
        scaling_factor = math.sqrt(std)
        dataset_[feature] = dataset_[feature] / scaling_factor
    for col in dropped_columns:
        dataset_ = pd.concat([dataset_, col], axis=1)
    return dataset_


def concatenate_data(
        lipidomics, metabolomics, proteomics,
        resample_train=False, resample_test=False, num_classes=3):
    """
    Load the concatenated dataset and split it into training and testing set
    :param lipidomics: the lipidomics dataset
    :param metabolomics: the metabolomics dataset
    :param proteomics: the proteomics dataset
    :param resample_train: whether to resample the training set
    :param resample_test: whether to resample the testing set
    :param num_classes: 3 for the multiclass classification, 2 for the binary classification
    :return:
    """

    # Merge the datasets on RID
    integrated_dataset = pd.merge(lipidomics, metabolomics, on="RID", how="inner")
    integrated_dataset = pd.merge(integrated_dataset, proteomics, on="RID", how="inner")

    # Get the train and test split
    train, test = get_train_test_split_dataset(integrated_dataset)

    # Get the class labels
    if num_classes == 2:
        Y_train_, Y_test_ = train["TwoClass"], test["TwoClass"]
    else:
        Y_train_, Y_test_ = train["ThreeClass"], test["ThreeClass"]

    # Drop columns that have following substrings in their names
    columns_to_drop = ["ThreeClass", "TwoClass", "RID"]
    for col in train.columns:
        if columns_to_drop[0] in col or columns_to_drop[1] in col or columns_to_drop[2] in col:
            train = train.drop(col, axis=1)
            test = test.drop(col, axis=1)

    if resample_train:
        train, Y_train_ = resampling(train, Y_train_)
    if resample_test:
        test, Y_test_ = resampling(test, Y_test_)
    return train, test, Y_train_, Y_test_


def load_data():
    """
    Load the lipidomics, metabolomics and proteomics datasets from the predefined path
    """
    path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset_split.xls"
    xls = pd.ExcelFile(path)
    lipidomics = pd.read_excel(xls, "Lipidomics")
    metabolomics = pd.read_excel(xls, "Metabolomics")
    proteomics = pd.read_excel(xls, "Proteomics")
    return lipidomics, metabolomics, proteomics


if __name__ == "__main__":
    l, m, p = load_data()
    X_train, X_test, Y_train, y_test = concatenate_data(l, m, p, resample_train=True, resample_test=False)
    substr = "ThreeClass"
    cols = X_train.columns
    # Get the name of the features where substr is in the name
    print([col for col in cols if substr in col])
    # print(Y_train)
