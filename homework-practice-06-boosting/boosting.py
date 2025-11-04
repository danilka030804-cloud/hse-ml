from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            depth: int = None,
            min_sample_split: int = 2,
            min_sample_leaf: int = 1,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.depth = depth
        self.sample_split = min_sample_split
        self.sample_leaf = min_sample_leaf

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []
        
        self.classes_ = [0, 1]

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        model = self.base_model_class(max_depth=self.depth, min_samples_split=self.sample_split, min_samples_leaf=self.sample_leaf)
        Y = y - predictions
        arr = np.random.choice(x.shape[0], size=int(self.subsample*x.shape[0]), replace=False)
        model.fit(x[arr], Y[arr])
        gamma = self.find_optimal_gamma(y, predictions, model.predict(x))
        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid = None, y_valid = None):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        if x_valid is not None:
            valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions = self.predict_proba(x_train)[:,1]
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['train_score'].append(self.score(x_train, y_train))
            if x_valid is not None:
                valid_predictions = self.predict_proba(x_valid)[:,1]
                self.history['valid_loss'].append(self.loss_fn(y_valid, valid_predictions))
                self.history['valid_score'].append(self.score(x_valid, y_valid))

            if self.early_stopping_rounds is not None:
                break

        if self.plot:
            fig, ax = plt.subplots(ncols=2)
            ax[0].plot(np.arange(self.n_estimators), self.history['train_loss'], label='train')
            if x_valid is not None:
                ax[0].plot(np.arange(self.n_estimators), self.history['valid_loss'], label='valid')
            ax[0].set_xlabel('N estimators')
            ax[0].set_ylabel('loss')
            ax[0].set_title('Loss')
            
            ax[1].plot(np.arange(self.n_estimators), self.history['train_score'], label='train')
            if x_valid is not None:
                ax[1].plot(np.arange(self.n_estimators), self.history['valid_score'], label='valid')
            ax[1].set_xlabel('N estimators')
            ax[1].set_ylabel('Score')
            ax[1].set_title('AUC-ROC')
            
            plt.legend()
            plt.grid(True)
            plt.show()

    def predict_proba(self, x):
        logit = 0
        for gamma, model in zip(self.gammas, self.models):
            logit += gamma * model.predict(x)
        prob = np.column_stack((1-self.sigmoid(logit), self.sigmoid(logit)))
        return prob

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        imp = [model.feature_importances_ for model in self.models]
        imp = np.mean(imp, axis=0)
        imp /= np.sum(imp)
        return imp
    
    def get_params(self, deep=True):
        """Обязательный метод для sklearn"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            # 'gamma': self.gammas
        }
    
    def set_params(self, **params):
        """Обязательный метод для sklearn"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def predict(self, x):
        prob = self.predict_proba(x)
        dec = np.argmax(prob, axis=0)
        return dec