from scipy.stats import randint
import pandas as pd
from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from helper import get_train_test_split_dataset
import graphviz
from IPython.display import Image
from sklearn.tree import export_graphviz
from imblearn.over_sampling import SMOTE


def resampling(x_train, y_train):
    """
    Resample the training set to balance the classes

    Initial dataset is very imbalanced, with the majority class being the "Stable" class
    Therefore the tree always predicted the majority class
    """
    smote = SMOTE()
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    print("Resampled training set shape: ", x_train_resampled.shape)
    print("Resampled training set class distribution: \n", y_train_resampled.value_counts())
    return x_train_resampled, y_train_resampled


def plot_decision_trees(rf, x_train, num_trees=3):
    for i in range(num_trees):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(
            tree,
            feature_names=x_train.columns,
            class_names=["Stable", "Slow Decliner", "Fast Decliner"],
            filled=True, rounded=True,
            max_depth=4,
            impurity=False,
            proportion=True)
        graph = graphviz.Source(dot_data, directory="trees")
        graph.render(f'tree_{i}', format='png')
        display(Image(f'tree_{i}.png'))


def plot_confusion_matrix(y_test, y_prediction):
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_prediction)
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def random_forest(x_train, y_train, tune_hyperparameters=True):
    """
    Constructs multiple decision trees during training time and outputs the class
    that is the mode of the classes of the individual trees

    Adapted from:
    https://www.datacamp.com/tutorial/random-forests-classifier-python
    :param x_train: Training set features
    :param y_train: Training set labels
    :param tune_hyperparameters: Whether to tune the hyperparameters
    :return:
    """

    # Hyperparameter tuning
    param_dist = {
        # Number of decision trees in the forest.
        # Generally, the higher, the better performance but it increases the computation cost
        'n_estimators': randint(200, 500),
        # The maximum depth of each tree in the forest
        # Higher values could lead to overfitting, while lower values could lead to underfitting
        'max_depth': randint(5, 15)}

    rf = RandomForestClassifier(
        class_weight="balanced",
    )

    if not tune_hyperparameters:
        # Fit the model to the training data
        rf.fit(x_train, y_train)
        return rf

    else:
        # Perform randomized search on hyperparameters
        rand_search = RandomizedSearchCV(
            estimator=rf, param_distributions=param_dist,
            n_iter=10, cv=5)

        # Fit the model to the training data
        rand_search.fit(x_train, y_train)

        # Best parameters
        best_params = rand_search.best_params_
        print("Best parameters:", best_params)

        # Best model
        best_rf = rand_search.best_estimator_
        return best_rf


def concatenation_integration(lipidomics_, metabolomics_, proteomics_):
    """
    Concatenate the datasets and integrate them
    TODO check whether dataset scaling relative to its size is needed
    :param lipidomics_: the lipidomics dataset
    :param metabolomics_: the metabolomics dataset
    :param proteomics_: the proteomics dataset
    :return: the integrated dataset
    """
    # Merge the datasets
    integrated_dataset = pd.merge(lipidomics_, metabolomics_, on="RID", how="inner")
    integrated_dataset = pd.merge(integrated_dataset, proteomics_, on="RID", how="inner")
    print("Integrated dataset shape: ", integrated_dataset.shape)
    train, test = get_train_test_split_dataset(integrated_dataset)

    X_train, X_test = train.iloc[:, 3:], test.iloc[:, 3:]
    Y_train, y_test = train['ThreeClass'], test['ThreeClass']
    return X_train, X_test, Y_train, y_test


if __name__ == "__main__":
    # Load the dataset
    path = "C:/Users/mjgoj/Desktop/THESIS/data/final_dataset_split.xls"
    xls = pd.ExcelFile(path)

    lipidomics = pd.read_excel(xls, "Lipidomics")
    metabolomics = pd.read_excel(xls, "Metabolomics_Pareto_Scaled")
    proteomics = pd.read_excel(xls, "Proteomics_Pareto_Scaled")

    X_train_, X_test_, Y_train_, y_test_ = concatenation_integration(lipidomics, metabolomics, proteomics)

    # Resample the training set
    X_train_resampled, Y_train_resampled = resampling(X_train_, Y_train_)
