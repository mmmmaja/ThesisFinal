import math
import os

import seaborn as sns
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    f1_score, roc_curve, auc, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

"""
This file contains helper methods and plotting methods that are used in the main notebooks
"""

# Load the data (it will be used by different functions)
dataset_path = os.path.join('..', 'data', 'final_dataset_split.xls')
xls = pd.ExcelFile(dataset_path)

lipidomics = pd.read_excel(xls, "Lipidomics")
metabolomics = pd.read_excel(xls, "Pareto Metabolomics")
proteomics = pd.read_excel(xls, "Pareto Proteomics")

metabolomics_unscaled = pd.read_excel(xls, "Metabolomics")
proteomics_unscaled = pd.read_excel(xls, "Proteomics")

# Define the color palette for the omics types
colors = sns.color_palette("husl", 3)
OMICS_PALETTE = {
    "Proteomics": colors[0],
    "Metabolomics": colors[1],
    "Lipidomics": colors[2]
}

# Read the conversion path for the metabolite and protein names
protein_conversion_path = os.path.join('..', 'data', 'proteomics_conversion.xlsx')
protein_conversion = pd.read_excel(protein_conversion_path, sheet_name="Transitions")

metabolite_conversion_path = os.path.join('..', 'data', 'metabolomics_conversion.xlsx')
metabolite_conversion = pd.read_excel(metabolite_conversion_path, sheet_name="Metabolomics")
# Add new column for the metabolite name in the conversion table
metabolite_conversion["UPPER_NAME"] = metabolite_conversion["NAME"].str.upper()


# PLOTTING METHODS


def plot_confusion_matrix(y_test_, y_prediction_, vmax=None):
    """
    Plot the confusion matrix of the model with consistent colorbar limits and correct color mapping
    :param y_test_: True labels
    :param y_prediction_: Predicted labels
    :param vmax: Maximum value for colormap scaling
    (needed for consistent colorbar and for the plots to be comparable)
    """

    # Get unique labels from the test and prediction datasets
    labels = np.unique(np.concatenate([y_test_, y_prediction_]))
    # Map numerical labels to custom labels
    if len(labels) == 3:
        label_order = [0, 1, 2]
        label_mapping = {0: 'Stable', 1: 'Slow Decliner', 2: 'Fast Decliner'}
    else:
        label_order = [0, 1]
        label_mapping = {0: 'Stable', 1: 'Decliner'}

    # Map the labels to the custom labels
    mapped_labels = [label_mapping[label] for label in label_order]

    # Create the confusion matrix
    cm = confusion_matrix(y_test_, y_prediction_, labels=labels)

    if vmax is None:
        vmax = cm.max()
    # Plot using matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu, vmin=0, vmax=vmax)

    # Adding color bar
    plt.colorbar(im, ax=ax)

    # Adding annotations
    thresh = cm.max() // 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > thresh:
                # First color in the colormap
                text_color = plt.cm.GnBu(0)
            else:
                # Last color in the colormap
                text_color = plt.cm.GnBu(256)
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center", color=text_color)

    ax.set_xticks(np.arange(len(mapped_labels)))
    ax.set_xticklabels(mapped_labels, rotation='horizontal')
    ax.set_yticks(np.arange(len(mapped_labels)))
    ax.set_yticklabels(mapped_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


def plot_ROC_curve(y_pred, y_test):
    """
    Plot the ROC curve of the model
    :param y_pred: Predicted labels
    :param y_test: True labels
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance(coefficients, features, n=10, omics=None):
    """
    :param coefficients: The coefficients determining feature importance
    :param features: The names of the features
    :param n: Number of top features to plot
    :param omics: The omics type of the features
    (Lipidomics, Metabolomics, Proteomics)
    """

    feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})
    if omics is not None:
        feature_importance = feature_importance[feature_importance['Omics'] == omics]
    else:
        feature_importance['Omics'] = [get_omics_type(feature) for feature in features]

    # Add the absolute importance
    feature_importance['Absolute Importance'] = abs(feature_importance['Importance'])
    # Sort the features based on the absolute value of the coefficient
    feature_importance = feature_importance.sort_values('Absolute Importance', ascending=False)

    # Select the top n features
    feature_importance = feature_importance[:n]

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance', y='Feature', hue='Omics',
        data=feature_importance, palette=OMICS_PALETTE, legend=True
    )
    # Add the legend
    plt.legend(title='Omics')

    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Importance', fontsize=16)
    plt.show()


# HELPER METHODS

def get_protein_name(peptide_sequence):
    """
    Get the protein name using the peptide sequence (conversion is defined in the protein_conversion table)
    :param peptide_sequence: Peptide sequence that corresponds to the protein
    :return: Protein name
    """
    # Get the protein name
    row = protein_conversion.loc[protein_conversion["PeptideSequence"] == peptide_sequence]
    if row.empty:
        return None
    else:
        return row["Protein"].values[0]


def get_metabolite_name(metabolite_id):
    """
    Get the metabolite name from the metabolite ID (conversion is defined in the metabolite_conversion table)
    :param metabolite_id: unique identifier of the metabolite
    :return: Metabolite name
    """
    # Map . to _ in the name
    metabolite_id = metabolite_id.replace(".", "_")
    # Strip from the right
    metabolite_id = metabolite_id.rstrip("_").upper()
    # Since the conversion table has some special cases, we need to define a conversion switch
    conversion_switch = {
        "TG_PG": "TG_BY_PG",
        "APOB_APOA1": "APOB_BY_APOA1",
        "GLOL": "GLYCEROL",
        "MUFA_FA": "MUFA_PCT",
        "PUFA_FA": "PUFA_PCT",
        "SFA_FA": "SFA_PCT",
        "TOTFA": "TOTAL_FA",
        "DHA_FA": "DHA_PCT",
        "LA_FA": "LA_PCT",
        "GLC": "GLUCOSE",
        "VLDL_D": "VLDL_SIZE",
    }
    # Check if the metabolite_id is in the conversion switch (It includes special cases)
    if metabolite_id in conversion_switch:
        metabolite_id = conversion_switch[metabolite_id]
    # Get the metabolite name from the conversion table
    row_index = metabolite_conversion["UPPER_NAME"].str.contains(metabolite_id)
    row = metabolite_conversion[row_index]
    if row.empty:
        return "Unknown"
    else:
        return row["LABEL"].values[0]


def map_lipid_name(lipid_name):
    """
    Map the lipid name to the LipidMaps format
    :param lipid_name: Name of the lipid
    """
    # Special cases
    if lipid_name.startswith("ACYLCARNITINE"):
        lipid_name = lipid_name.replace("ACYLCARNITINE", "CAR")
    if lipid_name.startswith("HEXCER"):
        lipid_name = lipid_name.replace("HEXCER", "Glc-Cer")
    if lipid_name.startswith("HEX2CER"):
        lipid_name = lipid_name.replace("HEX2CER", "Lac-Cer")
    if lipid_name.startswith("HEX3CER"):
        lipid_name = lipid_name.replace("HEX2CER", "Lac-Cer")

    # first _ is mapped to (
    lipid_name = lipid_name.replace("_", "(", 1)
    # last _ is mapped to )
    lipid_name = lipid_name[::-1].replace("_", ")", 1)[::-1]
    # get number of _ left
    num_ = lipid_name.count("_")
    if num_ == 2:
        lipid_name = lipid_name.replace("_", "-", 1)
        lipid_name = lipid_name.replace("_", ":", 1)
    if num_ == 4:
        lipid_name = lipid_name.replace("_", "-", 1)
        num_ -= 1
    if num_ == 3:
        # second _ is mapped to /
        lipid_name = lipid_name.replace("_", ":", 1)
        lipid_name = lipid_name.replace("_", "/", 1)
        lipid_name = lipid_name.replace("_", ":", 1)
    elif num_ == 1:
        lipid_name = lipid_name.replace("_", ":", 1)

    if num_ == 9:
        lipid_name = lipid_name.replace("_", ":", 1)
        lipid_name = lipid_name.replace("_", "/", 1)
        lipid_name = lipid_name.replace("_", ":", 1)
        lipid_name = lipid_name.replace("_", ")", 1)
        lipid_name = lipid_name.replace("_", ",", 1)
        lipid_name = lipid_name.replace("_", "(", 1)
        lipid_name = lipid_name.replace("_", ":", 1)
        lipid_name = lipid_name.replace("_", "/", 1)
        lipid_name = lipid_name.replace("_", ":", 1)

    return lipid_name


def get_omics_type(feature_name):
    """
    Get the omics type of the feature
    :param feature_name: Name of the feature
    :return: Omics type of the feature
    """
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
    columns = X.columns
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=columns)
    return X


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
    print(f'Weighted F1 score of the model: {f1:.4f}')
    MCC = matthews_corrcoef(y_true, y_pred)
    print(f'Matthews Correlation Coefficient of the model: {MCC:.4f}')


def get_train_test_split_RIDs():
    """
    Get the RIDs of the training and testing set from the file
    :return: the RIDs of the training and testing set
    """
    RID_path = os.path.join('..', 'data', 'RIDs.txt')
    with open(RID_path, "r") as file:
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
    return training_set, testing_set


def resampling(x, y):
    """
    Resample the dataset to balance the classes
    The Initial dataset is very imbalanced, with the majority class being the "Stable" class,
    Therefore, the tree always predicted the majority class
    :param x: Features
    :param y: Labels
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
    :param num_classes: 3 for the multiclass classification, 2 for the binary_classification classification
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


def load_data(scaled=True):
    """
    Load the data
    :return: the lipidomics, metabolomics, and proteomics datasets
    """
    if scaled:
        return lipidomics, metabolomics, proteomics
    else:
        return lipidomics, metabolomics_unscaled, proteomics_unscaled
