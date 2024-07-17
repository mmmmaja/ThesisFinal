from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from methods.feature_selection import mutual_information_filtering
from abc import ABC, abstractmethod
import os
import sys
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, log_loss


def plot_filtering_tuning(results, metric):
    """
    Plot the results of the feature selection tuning
    :param results: dictionary containing the results of the tuning
    :param metric: the metric used for evaluation
    """

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(8, 6))

    plt.plot(
        results.keys(), results.values(),
        marker='o', linestyle='dashed', linewidth=1, markersize=6
    )
    # Plot the best result
    best = max(results, key=results.get)
    plt.plot(best, results[best], 'ro')

    plt.xlabel('Keep ratio')
    plt.ylabel(metric)
    plt.title('Feature selection tuning')
    plt.show()


class BaseFilter(ABC):

    def __init__(self, tune_keep_ratio=False, keep_ratio=0.8):
        self.tune_keep_ratio, self.keep_ratio = tune_keep_ratio, keep_ratio

    def get_features(self, X, y, model):
        if self.tune_keep_ratio:
            self.keep_ratio = self.tune_filter(X, y, model)
            print(f"Selected keep ratio: {self.keep_ratio}")
        return self.filter(X, y)

    @abstractmethod
    def tune_filter(self, X, y, model):
        pass

    @abstractmethod
    def filter(self, X, y):
        pass


def get_score(metric, Y_test, binary_prediction, continuous_prediction):
    if metric == 'f1_weighted':
        return f1_score(Y_test, binary_prediction, average='weighted')
    elif metric == 'f1':
        return f1_score(Y_test, binary_prediction, average='binary')
    elif metric == 'roc_auc':
        return roc_auc_score(Y_test, continuous_prediction)
    elif metric == 'matthews_corrcoef':
        return matthews_corrcoef(Y_test, binary_prediction)
    elif metric == 'neg_log_loss':
        return -log_loss(Y_test, continuous_prediction)
    else:
        return accuracy_score(Y_test, binary_prediction)


class MIFSFilter(BaseFilter):

    def __init__(self, tune_keep_ratio=False, keep_ratio=0.8):
        super().__init__(tune_keep_ratio, keep_ratio)

    def filter(self, X, y):
        selected_features = mutual_information_filtering(X, y, self.keep_ratio)
        return selected_features

    def tune_filter(self, X, y, model, n=10):
        # Split into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        # Resample the training data using SMOTE
        X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

        # Dictionary to store the results
        results = {}
        # Iterate over the keep ratios
        for k_candidate in tqdm(range(1, n + 1), desc="Tuning MIFS"):
            keep_ratio = k_candidate / n
            # Get the selected features
            features = mutual_information_filtering(X_train, Y_train, keep_ratio)
            # Fit the model using the filtered data
            # Silence the output from the terminal here
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    model.fit(X_train[features], Y_train)
                finally:
                    sys.stdout = old_stdout

            # Make predictions
            binary_predictions, continuous_predictions = model.predict(X_test[features])
            results[keep_ratio] = get_score(model.scoring, Y_test, binary_predictions, continuous_predictions)

        plot_filtering_tuning(results, model.scoring)
        best_keep_ratio = max(results, key=results.get)
        print(f"Best keep ratio: {best_keep_ratio}, best score: {results[best_keep_ratio]:.3f}")
        return best_keep_ratio
