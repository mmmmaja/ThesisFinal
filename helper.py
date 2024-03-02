import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

"""
This file consists of helper methods for feature selection
as well as Ven diagrams for visualizing the intersection of features (3 or 2 sets)
and a method for getting the training and testing set from the dataset
"""


def ANOVA_correlation_filtering(k, X_, Y_):
    """
    ANOVA F-value statistics to select the top k features
    that have the highest correlation to the target variable
    :param k: the number of features to select
    :param X_: the feature matrix
    :param Y_: the class labels
    """
    selector = SelectKBest(f_classif, k=k)  # Selects the top 10 features
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    print("Number of features after ANOVA filtering ", len(features))
    return features


def variance_filtering(X_, threshold=0.01):
    """
    Removing features with low variance.
    It is the only non-supervised feature selection method. It is important as it prevents overfitting
    :param X_: the feature matrix
    :param threshold: the threshold for variance
    """
    sel = VarianceThreshold(threshold=threshold)
    sel.fit_transform(X_)
    # Return the names of the columns that were picked
    features = X_.columns[sel.get_support()]
    print("Number of features after variance filtering ", len(features))
    return features


def mutual_information_filtering(k, X_, Y_):
    """
    Mutual Information based Feature Selection (MIFS)
    Mutual information statistics to select the top k features
    that have the highest correlation to the target variable

    Measures how much information the presence/absence of a feature
    contributes to making the correct prediction on Y

    :param k: the number of features to select
    :param X_: the feature matrix
    :param Y_: the class labels
    """

    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    print("Number of features after mutual information filtering ", len(features))
    return features


def recursive_feature_elimination(k, X_, Y_, estimator=DecisionTreeClassifier()):
    """
    Wrapper approach to feature selection
    Starts with all features and removes the least important one at each iteration
    :param k: number of features to select
    :param X_: the feature matrix
    :param Y_: the class labels
    :param estimator: the model used to evaluate the importance of features
        DecisionTreeClassifier
        RandomForestClassifier

    """

    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    print("Number of features after recursive feature elimination ", len(features))
    return features


def venn_diagram(set_lists, set_labels, title=None):
    """
    Create a Venn diagram for 2 or 3 sets
    :param set_lists: list of sets of features
    :param set_labels: list of the filtering methods used
    :param title: the title of the plot
    """
    if len(set_lists) == 2:
        venn2(set_lists, set_labels)
    elif len(set_lists) == 3:
        venn3(set_lists, set_labels)
    if title:
        plt.title(title)
    plt.show()


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


if __name__ == "__main__":
    xls = pd.ExcelFile("C:/Users/mjgoj/Desktop/THESIS/data/final_dataset_split.csv")
    metabolomics = pd.read_excel(xls, "Metabolomics")
    get_train_test_split_dataset(metabolomics)