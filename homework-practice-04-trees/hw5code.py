import numpy as np
from collections import Counter
import random
import pandas as pd


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
    разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # Сортируем
    order = np.argsort(feature_vector)
    x_sorted = feature_vector[order]
    y_sorted = target_vector[order]

    # Убираем дубликаты соседних значений
    unique_vals = np.unique(x_sorted)
    if len(unique_vals) == 1:
        return np.array([]), np.array([]), None, np.inf

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

    n = len(y_sorted)
    cum_pos = np.cumsum(y_sorted)
    total_pos = cum_pos[-1]

    # вычисляем ginis для всех порогов
    ginis = []
    for t in thresholds:
        left_mask = x_sorted < t
        right_mask = ~left_mask
        n_left = left_mask.sum()
        n_right = right_mask.sum()

        p_left = y_sorted[left_mask].mean() if n_left > 0 else 0
        p_right = y_sorted[right_mask].mean() if n_right > 0 else 0

        H_left = 1 - p_left**2 - (1 - p_left)**2
        H_right = 1 - p_right**2 - (1 - p_right)**2

        g = -(n_left/n)*H_left - (n_right/n)*H_right
        ginis.append(g)

    ginis = np.array(ginis)
    best_idx = np.argmin(ginis)
    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]



class DecisionTree:
    def __init__(self, feature_types=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):

        if any(ft not in ("real", "categorical", None) for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        node['depth'] = depth
        if isinstance(sub_X, pd.DataFrame):
            sub_X = sub_X.to_numpy()
            
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if (self._min_samples_split is not None and len(sub_y) < self._min_samples_split) \
        or (self._max_depth is not None and node.get("depth", 0) >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y[:] == 1,feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_count / current_click
                    else:
                        current_click = 0
                        ratio[key] = 0
                    
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                feature_vector, uniques = pd.factorize(sub_X[:, feature])
                categories_map = {cat: idx for idx, cat in enumerate(uniques)}
            else:
                print(f"Unknown feature type at feature {feature}: {feature_type!r}, type={type(feature_type)}")
                print("All feature types:", self._feature_types)
                raise ValueError(f"Unknown feature type: {feature_type!r}")

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if (gini_best is None or gini < gini_best) & (threshold is not None):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if self._min_samples_leaf is not None:
                    n_left = np.sum(split)
                    n_right = np.sum(~split)
                    if n_left < self._min_samples_leaf or n_right < self._min_samples_leaf:
                        node["type"] = "terminal"
                        node["class"] = Counter(sub_y).most_common(1)[0][0]
                        return

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                            filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature] == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        """Обязательный метод для sklearn"""
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self