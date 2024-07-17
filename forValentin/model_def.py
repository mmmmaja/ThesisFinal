import os
from sklearn.preprocessing import OneHotEncoder
from abc import ABC, abstractmethod
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
import numpy as np
from methods.feature_selection import mutual_information_filtering
from methods.helper import standardize

os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.0"
os.environ["PATH"] = r"C:\Program Files\R\R-4.4.0\bin\x64" + ";" + os.environ["PATH"]
from rpy2.robjects import r


def fit_with_progress_bar(model, *args, **kwargs):
    """
    Fits the model and displays a progress bar
    Code adapted from:
    https://datascience.stackexchange.com/questions/114060/progress-bar-for-gridsearchcv
    :param model: ML model to fit
    :param args: the arguments to pass to the model
    :param kwargs: the keyword arguments to pass to the model
    :return: the fitted model
    """

    class BarStdout:
        def __init__(self):
            self.bar_size, self.bar, self.count, self.done = None, None, 0, False

        def write(self, text):
            if self.done:
                return
            # Initialize the progress bar
            elif "totalling" in text and "fits" in text:
                self.bar_size = int(text.split("totalling")[1].split("fits")[0][1:-1])
                self.bar, self.count = tqdm(range(self.bar_size)), 0
            # Update the progress bar
            elif "CV" in text and hasattr(self, "bar"):
                self.count += 1
                self.bar.update(n=self.count - self.bar.n)
                # Mark as done when the progress bar completes
                if self.count >= self.bar_size:
                    self.done = True
                    self.bar.close()

        def flush(self, text=None):
            pass

    default_stdout = sys.stdout
    sys.stdout = BarStdout()
    try:
        # Set model verbosity to ensure it outputs progress information
        model.verbose = 10
        # Fit the model with the provided arguments
        model.fit(*args, **kwargs)
    finally:
        # Restore standard output to its original state
        sys.stdout = default_stdout

    return model


class BaseModel(ABC):
    """
    An abstract class representing a model.
    All the models should inherit from this class.
    """

    def __init__(self, seed, tune, scoring):
        """
        Initialize the model with the seed.
        """
        self.seed, self.tune = seed, tune
        self.model = None
        self.scoring = scoring

    def fit(self, x_train, y_train):
        """
        Fit the model to the training data.
        :param x_train: training features
        :param y_train: training labels
        :return: fitted model
        """
        if self.tune:
            self.tune_model(x_train, y_train)
        self.model.fit(x_train, y_train)

    def predict(self, x):
        """
        Predict the labels for the given features.
        :param x: features
        :return: predicted labels (binary, continuous)
        """
        binary_predictions = self.model.predict(x)
        continuous_predictions = self.model.predict_proba(x)[:, 1]
        return binary_predictions, continuous_predictions

    @abstractmethod
    def tune_model(self, x_train, y_train, silent=False):
        """
        Tune the model to find the best hyperparameters.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Convert the model to a string.
        :return: the string representation of the model
        """
        pass

    @abstractmethod
    def deep_copy(self):
        """
        Create a copy of the model.
        :return: the copied model
        """
        pass


class LogisticRegressionModel(BaseModel):

    def __init__(self, seed=42, c=1, solver='saga', tune=False, scoring='f1_weighted'):
        super().__init__(seed, tune, scoring)
        self.model = LogisticRegression(
            max_iter=1000, multi_class='ovr', class_weight='balanced',
            penalty='l1', solver=solver, C=c, random_state=self.seed
        )

    def fit(self, train_x, train_y):
        # Standardize the data
        train_x = standardize(train_x)
        if self.tune:
            self.tune_model(train_x, train_y)
        # Fit the model
        self.model.fit(train_x, train_y.values.ravel())

    def tune_model(self, train_x, train_y, silent=True):
        """
        Tune the model using GridSearchCV to find the best hyperparameters.
        """
        # Standardize the data
        train_x = standardize(train_x)
        hyperparameters = {
            'C': [0.01, 0.1, 0.3, 0.7, 1, 3, 5, 10, 20, 50]
        }
        grid_search = GridSearchCV(
            LogisticRegression(
                max_iter=1000, multi_class='ovr', class_weight='balanced',
                penalty='l1', solver='saga', random_state=self.seed
            ),
            hyperparameters, cv=5, scoring=self.scoring
        )
        if silent:
            grid_search.fit(train_x, train_y.values.ravel())
        else:
            fit_with_progress_bar(grid_search, train_x, train_y.values.ravel())
        self.model = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)

    def predict(self, x):
        # Standardize the data
        x = standardize(x)
        binary_predictions = self.model.predict(x)
        continuous_predictions = self.model.predict_proba(x)[:, 1]
        return binary_predictions, continuous_predictions

    def __str__(self):
        return f"LogisticRegression(C={self.model.C}, solver={self.model.solver}, score={self.scoring})"

    def deep_copy(self):
        return LogisticRegressionModel(
            self.seed, self.model.C, self.model.solver, self.tune, self.scoring
        )


class AdaBoostModel(BaseModel):

    def __init__(self, seed=42, learning_rate=0.01, n_estimators=400, tune=False, scoring='f1_weighted'):
        """
        Initialize the AdaBoost model.
        :param seed: the random seed
        :param learning_rate: How much the contribution of each classifier is reduced.
        :param n_estimators: The maximum number of estimators at which boosting is terminated.
        """
        super().__init__(seed, tune, scoring)
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            learning_rate=learning_rate
        )

    def tune_model(self, train_x, train_y, silent=True):
        hyperparameters = {
            'n_estimators': [50, 100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5]
        }
        grid_search = GridSearchCV(
            AdaBoostClassifier(random_state=self.seed),
            hyperparameters, cv=5, scoring=self.scoring)
        if silent:
            grid_search.fit(train_x, train_y.values.ravel())
        else:
            fit_with_progress_bar(grid_search, train_x, train_y.values.ravel())
        self.model = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)
        return grid_search.best_params_

    def __str__(self):
        return (f"AdaBoost(n_estimators={self.model.n_estimators}, "
                f"learning_rate={self.model.learning_rate}, score={self.scoring})")

    def deep_copy(self):
        return AdaBoostModel(
            self.seed, self.model.learning_rate, self.model.n_estimators,
            self.tune, self.scoring
        )


class RandomForestModel(BaseModel):

    def __init__(self,
                 seed=42, n_estimators=70, class_weight='balanced', scoring='f1_weighted',
                 min_samples_leaf=1, min_samples_split=2, max_depth=None, tune=False):
        super().__init__(seed, tune, scoring)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split, class_weight=class_weight
        )

    def tune_model(self, x_train, y_train, silent=True):
        hyperparameters = {
            'n_estimators': [50, 100, 200, 300, 400],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.seed, max_depth=12, class_weight='balanced'),
            hyperparameters, cv=5, scoring=self.scoring
        )
        if silent:
            grid_search.fit(x_train, y_train.values.ravel())
        else:
            fit_with_progress_bar(grid_search, x_train, y_train.values.ravel())
        self.model = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)
        return grid_search.best_params_

    def __str__(self):
        return (f"RandomForest(n_estimators={self.model.n_estimators}, "
                f"min_samples_split={self.model.min_samples_split}, "
                f"min_samples_leaf={self.model.min_samples_leaf}, "
                f"max_depth={self.model.max_depth}, score={self.scoring}, "
                f"class_weight={self.model.class_weight})")

    def deep_copy(self):
        return RandomForestModel(
            self.seed, self.model.n_estimators, self.model.class_weight,
            self.model.min_samples_leaf, self.model.min_samples_split, self.model.max_depth,
            self.tune, self.scoring
        )


class PLSDAModel(BaseModel):

    def __init__(self, n_components=4, seed=42, tune=False, scoring='f1_weighted'):
        super().__init__(seed, tune, scoring)
        self.model = PLSRegression(
            n_components=n_components, scale=False
        )
        self.encoder = OneHotEncoder(sparse=False)

    def tune_model(self, x_train, y_train, silent=True):
        hyperparameters = {'n_components': [3, 4, 5, 6, 7]}
        grid_search = GridSearchCV(
            PLSRegression(scale=False),
            hyperparameters, cv=5, scoring=self.scoring
        )
        if silent:
            grid_search.fit(x_train, y_train)
        else:
            fit_with_progress_bar(grid_search, x_train, y_train)
        self.model = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)
        return grid_search.best_params_

    def fit(self, x_train, y_train):
        # Standardize the data
        x_train = standardize(x_train)
        y_train = self.encoder.fit_transform(y_train.values.reshape(-1, 1))
        if self.tune:
            self.tune_model(x_train, y_train)
        self.model.fit(x_train, y_train)
        return self.model

    def predict(self, x):
        # Standardize the data
        x = standardize(x)
        predictions = self.model.predict(x)
        binary_predictions = np.argmax(predictions, axis=1)
        continuous_predictions = None
        return binary_predictions, continuous_predictions

    def __str__(self):
        return f"PLS-DA(n_components={self.model.n_components}, score={self.scoring})"

    def deep_copy(self):
        return PLSDAModel(self.model.n_components, self.seed, self.tune, self.scoring)


class DIABLOModel(BaseModel):

    def __init__(self, seed=42, tune=False, scoring='f1_weighted'):
        super().__init__(seed, tune, scoring)
        self.model = None

    def tune_model(self, x_train, y_train, silent=True):
        pass

    def fit(self, x_train, y_train):
        pass

    def predict(self, x):
        pass

    def __str__(self):
        pass

    def deep_copy(self):
        pass
