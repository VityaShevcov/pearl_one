import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


class Pearl:

    def __init__(self, estimator, list_of_vars: list,
                 target: str = 'flag', cv=KFold(), go_on: bool = False,
                 path_dict: str = 'rank_dict.json', path_meta: str = 'df_meta.csv'):
        """
        Feature selection based on iterative addition ranked vars one by one in model
        and analysis the roc_auc score in each fold in cv.
        pass

        Args:
            estimator: estimator object
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a fit method,
            and attribute feature_importances_

            list_of_vars (list): list with vars to be ranked

            target (str): target var

            cv (optional): cross-validation generator. Determines the cross-validation splitting strategy.
            Defaults to KFold() from sklearn.model_selection.

            go_on (bool, optional): If the algo stops, you can continue feature selection with meta data.
            Defaults to False.

            path_dict (str, optional): Path to dict with ranked vars. If go_on = True - path to existing dict.
            Defaults to "rank_dict.json".

            path_meta (str, optional): Path to pandas dataframe with meta info.
            If go_on = True - path to existing meta info.
            Defaults to 'df_meta.csv'.
        """
        # list of vars to check the scores
        self.model_in = []
        # list of vars which have not been checked
        self.model_not_check = []

        self.estimator = estimator
        self.list_of_vars = list_of_vars
        self.target = target
        self.cv = cv
        self.go_on = go_on
        self.path_dict = path_dict
        self.path_meta = path_meta

    def get_features_rank(self, dataframe: pd.DataFrame, get_fe: bool = False) -> dict | tuple[dict, dict]:
        """
        Getting rank for all vars from list_of_vars by estimator.feature_importances_.
        The rank is calculated as sum of all feature_importances_ by all folds in cv
        Args:
            dataframe (pd.DataFrame): dataframe with all vars from list_of_vars and target

            get_fe (bool, optional): get dict with feature importances.
            If True the func returns two dict: rank_dict, fe_dict. Defaults to False

        Returns:
            dict: dict with var as key and rank: int as rank. Optional returns two dicts(see get_fe parameter)
        """

        dict_fold_importances = {'Feature': self.list_of_vars, 'fold': np.zeros(len(self.list_of_vars))}

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(dataframe), 1):
            train, val = dataframe.iloc[train_idx], dataframe.iloc[val_idx]
            self.estimator.fit(train[self.list_of_vars], train[self.target].values)

            # the more sum of feature_importances_ the more rank of it feature
            # (first rank = max(sum feature_importances_))
            dict_fold_importances['fold'] += self.estimator.feature_importances_

        # get dict of estimates as sum feature_importances_ for all folds
        fe_dict = {key: value for key, value in
                   zip(dict_fold_importances['Feature'], dict_fold_importances['fold'])}

        # get dict with ranked all vars by sum of feature_importances_
        rank_dict = {key: rank for rank, key in enumerate(sorted(fe_dict, key=fe_dict.get, reverse=True), 1)}

        if get_fe:
            return rank_dict, fe_dict

        return rank_dict

    def get_cv_scores(self, dataframe: pd.DataFrame, oof: np.array, get_train_scores: bool = False,
                      train_preds: np.array = None) -> tuple[np.array, np.array] | tuple[np.array, np.array, np.array]:
        """
        Getting out of fold (oof) and optional mean train predictions (train_preds)
        by model with including vars

        Args:
            dataframe (pd.DataFrame): sample for feature selection
            oof (np.array): out of fold predictions
            get_train_scores (bool, optional): option for getting mean train predictions.
            Defaults to False.
            train_preds (np.array, optional): Array for writing mean train prediction. Defaults to None.

        Returns:
            tuple[np.array, np.array] | tuple[np.array, np.array, np.array]: returns roc_auc_score
            for oot prediction on each fold (fold_auc) with its prediction values.
            Optionally with mean training prediction values.
        """

        fold_auc = np.zeros(self.cv.n_splits)

        # starting evaluating
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(dataframe), 1):
            train, val = dataframe.iloc[train_idx], dataframe.iloc[val_idx]
            self.estimator.fit(train[self.model_in], train[self.target].values)

            # writing predict probas
            oof[val_idx] = self.estimator.predict_proba(val[self.model_in])[:, 1]
            fold_auc[fold - 1] = round(roc_auc_score(val[self.target], oof[val_idx]), 4)

            # add train_preds for return if get_train_scores == True
            if get_train_scores:
                train_preds[train_idx] += \
                    self.estimator.predict_proba(train[[self.model_in]])[:, 1] / (self.cv.n_splits - 1)
                return fold_auc, oof, train_preds

        return fold_auc, oof
    
    @staticmethod
    def status_bar(var_name: str, len_of_spaces: str, iteration_index: int, lenj: int):
        """
        For getting status of iteration
        Args:
            var_name (str): var rank from the ranc_dict
            len_of_spaces (str): auxiliary parameter for sys.stdout
            iteration_index (int):auxiliary parameter for sys.stdout
            lenj (int): len of rank_dict. For sys.stdout
        """
        g = iteration_index / lenj
        sys.stdout.write(f'\r{var_name} {len_of_spaces}{round(g * 100, 2)}%')
        sys.stdout.flush()

    def var_selection_meta(self, var_key: int, var_name: str, fold_auc: np.array, old_auc: np.array,
                           iteration_index: int, len_of_spaces: str, lenj: int,
                           backward: bool = True) -> tuple[pd.DataFrame, np.array]:
        """
        Check the main rule for including var in the model.

        Args:
            var_key (int): var rank from the ranc_dict
            var_name (str): var name from ranc_dict
            fold_auc (np.array): roc_auc for each validation fold
            old_auc (np.array): roc_auc for each validation fold for the past iteration
            iteration_index (int): num of iteration index. In straight passing it will match with rank of var
            len_of_spaces (str): auxiliary parameter for sys.stdout
            lenj (int): len of rank_dict. For sys.stdout
            backward (bool, optional): pass. Defaults to True.

        Returns:
            tuple[pd.DataFrame, np.array]: new_df_meta for adding to meta file, old_auc - updated by rule
        """

        # for adding in meta file
        df_folds_old = pd.DataFrame(old_auc.copy().reshape(1, self.cv.n_splits),
                                    columns=[y + '_old' for y in
                                             ['fold' + str(i) for i in range(1, self.cv.n_splits + 1)]])
        df_folds_old.index = [iteration_index]

        # checkout rule
        summa = sum(fold_auc - old_auc > 0) * 1 + sum(fold_auc - old_auc < 0) * -1
        if backward:
            if summa > 0 or all(fold_auc == old_auc):
                old_auc = fold_auc.copy()
                flag_drop = 1
            else:
                self.model_in.append(var_name)
                flag_drop = 0
        else:
            if summa > 0:
                old_auc = fold_auc.copy()
                flag_drop = 0
            else:
                flag_drop = 1
                self.model_in.remove(var_name)

        self.status_bar(var_name, len_of_spaces, iteration_index, lenj)
        
        self.model_not_check.remove(var_name)

        # add to meta file
        df_folds = pd.DataFrame(fold_auc.copy().reshape(1, self.cv.n_splits),
                                columns=['fold' + str(i) for i in range(1, self.cv.n_splits + 1)])
        df_folds.index = [iteration_index]
        new_df_meta = df_folds.join(df_folds_old)
        new_df_meta['var_rank'] = var_key
        new_df_meta['flag_drop'] = flag_drop

        return new_df_meta, old_auc

    def get_selected(self, dataframe: pd.DataFrame) -> list:
        """
        Returns final list with vars, writing all steps in meta file

        Args:
            dataframe (pd.DataFrame): sample for feature selection

        Returns:
            list: list with all vars, selected by rule
        """
        # for continuing feature selection with existing meta files
        if self.go_on:
            rank_dict = json.load(open(self.path_dict))
            df_meta = pd.read_csv(self.path_meta, index_col=0)
            self.model_in = [t[0] for t in rank_dict.items() \
                             if t[1] in df_meta[df_meta['flag_drop'] == 0]['var_rank'].values]
            max_rank = max(df_meta['var_rank'])
            self.model_not_check = [i for i in rank_dict.keys() if rank_dict[i] > max_rank]
            iteration_index = max_rank
            if df_meta.iloc[-1, :]['flag_drop']:
                old_auc = np.array(df_meta.iloc[-1, :]['fold1_old': 'fold4_old'])
            else:
                old_auc = np.array(df_meta.iloc[-1, :]['fold1': 'fold4'])

        # for starting new feature selection process
        else:
            rank_dict = self.get_features_rank(dataframe)
            json.dump(rank_dict, open(self.path_dict, 'w'))
            self.model_not_check = [i for i in rank_dict.keys()]
            iteration_index = 0
            max_rank = 0

        # creating np.arrays for writing predictions
        oof = np.zeros(len(dataframe))
        # train_preds = np.zeros(len(dataframe))

        lenj = len(rank_dict)
        max_len = max([len(feature_col) for feature_col in self.model_not_check])
        df_meta = pd.DataFrame()
        old_auc = []
        # starting feature selection
        for var_name, var_key in [i for i in rank_dict.items()][max_rank:]:

            len_of_spaces = (max_len - len(var_name)) * ' '
            iteration_index += 1

            if iteration_index == 1:
                self.model_in.append(var_name)

                fold_auc, oof = self.get_cv_scores(dataframe, oof)
                old_auc = fold_auc.copy()
                
                self.status_bar(var_name, len_of_spaces, iteration_index, lenj)
                
                self.model_not_check.remove(var_name)

                df_folds = pd.DataFrame(fold_auc.copy().reshape(1, self.cv.n_splits),
                                        columns=['fold' + str(i) for i in range(1, self.cv.n_splits + 1)])
                df_folds_old = pd.DataFrame(old_auc.copy().reshape(1, self.cv.n_splits),
                                            columns=[y + '_old' for y in
                                                     ['fold' + str(i) for i in range(1, self.cv.n_splits + 1)]])
                df_folds.index, df_folds_old.index = [iteration_index], [iteration_index]
                df_meta = df_folds_old.join(df_folds)
                df_meta['var_rank'] = var_key
                df_meta['flag_drop'] = 0

            else:
                self.model_in.append(var_name)
                fold_auc, oof = self.get_cv_scores(dataframe, oof)
                new_df_meta, old_auc = self.var_selection_meta(var_key, var_name,
                                                               fold_auc, old_auc, iteration_index, len_of_spaces, lenj,
                                                               backward=False)
                df_meta = pd.concat([df_meta, new_df_meta])

            # saving each step in meta file
            df_meta.to_csv(self.path_meta)

        return [t[0] for t in rank_dict.items() \
                             if t[1] in df_meta[df_meta['flag_drop'] == 0]['var_rank'].values]

    def backward(self):
        pass