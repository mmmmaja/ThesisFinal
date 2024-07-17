import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from venn import venn
from insignificant.buruta_py import BorutaPy

"""
This file consists of helper methods for feature selection
as well as Ven diagrams for visualizing the intersection of features (3 or 2 sets)
and a method for getting the training and testing set from the dataset
"""


def ANOVA_correlation_filtering(X_, Y_, keep_ratio=0.8):
    """
    ANOVA F-value statistics to select the top k features
    that have the highest correlation to the target variable
    :param keep_ratio: the ratio of features to keep
    :param X_: the feature matrix
    :param Y_: the class labels
    """

    # Get number of features to keep
    k = int(len(X_.columns) * keep_ratio)

    selector = SelectKBest(f_classif, k=k)
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    return features


def variance_filtering(X_, Y_=None, threshold=0.01):
    """
    Removing features with low variance.
    It is the only non-supervised feature selection method. It is important as it prevents overfitting
    :param X_: the feature matrix
    :param Y_: the class labels (Not used in this method)
    :param threshold: the threshold for variance
    """
    sel = VarianceThreshold(threshold=threshold)
    sel.fit_transform(X_)
    # Return the names of the columns that were picked
    features = X_.columns[sel.get_support()]
    # print("Number of features after variance filtering ", len(features))
    return features


def mutual_information_filtering(X_, Y_, keep_ratio=0.8):
    """
    Mutual Information based Feature Selection (MIFS)
    Mutual information statistics to select the top k features
    that have the highest correlation to the target variable

    Measures how much information the presence/absence of a feature
    contributes to making the correct prediction on Y

    :param keep_ratio: the ratio of features to keep
    :param X_: the feature matrix
    :param Y_: the class labels
    """

    # Get number of features to keep
    k = int(len(X_.columns) * keep_ratio)

    # Create the selector
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    # print("Number of features after mutual information filtering ", len(features))
    return features


def recursive_feature_elimination(X_, Y_, keep_ratio=0.8, estimator=DecisionTreeClassifier()):
    """
    Wrapper approach to feature selection
    Starts with all features and removes the least important one at each iteration
    :param keep_ratio: number of features to select
    :param X_: the feature matrix
    :param Y_: the class labels
    :param keep_ratio: the ratio of features to keep
    :param estimator: the model used to evaluate the importance of features
        DecisionTreeClassifier
        RandomForestClassifier
    """

    # Get number of features to keep
    k = int(len(X_.columns) * keep_ratio)

    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit_transform(X_, Y_)
    # Return the names of the columns that were picked
    features = X_.columns[selector.get_support()]
    # print("Number of features after recursive feature elimination ", len(features))
    return features


def boruta_filtering(X_, Y_, keep_ratio=0.8):
    """
    Boruta is an all-relevant feature selection method (no threshold is needed)
    Code inspired by:
    https://towardsdatascience.com/simple-example-using-boruta-feature-selection-in-python-8b96925d5d7a

    1)
    For each feature in the dataset, Boruta creates a shadow feature by shuffling the values of the
    original feature.
    This effectively breaks any relationship between the feature and the target.

    2)
    Boruta uses a random forest classifier to compute the importance of both original and shadow features.

    3)
    The importance of each original feature is compared to the highest importance among shadow features.
    """
    # Convert to matrix
    X = X_.values
    Y = Y_.values

    # Define the model
    # rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    rf = RandomForestClassifier()

    # Define the Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

    # Find all relevant features
    feat_selector.fit(X, Y)

    # Return the names of the columns that were picked
    return X_.columns[feat_selector.support_]


def venn_diagram(set_lists, set_labels, title=None, colors=None, alpha=0.4):
    """
    Create a Venn diagram for 2 or 3 sets
    :param set_lists: list of sets of features
    :param set_labels: list of the filtering methods used
    :param title: the title of the plot
    :param colors: the colors of the circles
    :param alpha: the transparency of the circles
    """
    if not colors:
        colors = sns.color_palette("Set2", len(set_lists))
    if len(set_lists) == 2:
        venn2(set_lists, set_labels, set_colors=colors, alpha=alpha)
    elif len(set_lists) == 3:
        venn3(set_lists, set_labels, set_colors=colors, alpha=alpha)
    elif len(set_lists) > 3:
        venn(set_lists, set_labels, set_colors=colors, alpha=alpha)
    if title:
        plt.title(title)
    plt.show()
