import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class Pearl(BaseEstimator, TransformerMixin):

    def __init__(self, vars_dict: dict):

        self.model_in = []
        self.model_drop = []
        self.model_not_check = [i for i in vars_dict.values()]

    def get_features_rank(self, estimator, dataframe: pd.DataFrame, list_of_vars: list, target: str, cv=KFold(),
                          get_fe: bool = False) -> Union[tuple[dict, dict], dict]:
        """
        Getting rank for all vars from list_of_vars by estimator.feature_importances_.
        The rank is calculated as sum of all feature_importances_ by all folds in cv
        Args:
            estimator: estimator object
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a fit method,
            and attribute feature_importances_

            dataframe (pd.DataFrame): dataframe with all vars from list_of_vars and target

            list_of_vars (list): list with vars to be ranked

            target (str): target var

            cv (optional): cross-validation generator. Determines the cross-validation splitting strategy.
            Defaults to KFold() from sklearn.model_selection.

            get_fe (bool, optional): get dict with feature importances.
            If True the func returns two dict: rank_dict, fe_dict. Defaults to False
        Returns:
            dict: dict with var as key and rank: int as rank. Optional returns two dicts(see get_fe parameter)
        """
        
        dict_fold_importances = {'Feature': list_of_vars, 'fold': np.zeros(len(list_of_vars))}

        for fold, (train_idx, val_idx) in enumerate(cv.split(dataframe), 1):
            train, val = dataframe.iloc[train_idx], dataframe.iloc[val_idx]
            estimator.fit(train[list_of_vars], train[target].values)

            # the more sum of feature_importances_ the more rank of it feature
            # (first rank = max(sum feature_importances_))
            dict_fold_importances['fold'] += estimator.feature_importances_

        # get dict of estimates as sum feature_importances_ for all folds
        fe_dict = {key: value for key, value in
                       zip(dict_fold_importances['Feature'], dict_fold_importances['fold'])}

        # get dict with ranked all vars by sum of feature_importances_
        rank_dict = {key: rank for rank, key in enumerate(sorted(fe_dict, key=fe_dict.get, reverse=True), 1)}

        if get_fe:
            return rank_dict, fe_dict

        return rank_dict