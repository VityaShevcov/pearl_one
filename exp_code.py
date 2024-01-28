import pandas as pd
import numpy as np
import pickle
import shap
from typing import List, Callable, Optional, Tuple, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

class DateTimeSeriesSplit:
    def __init__(self, n_splits: int = 4, test_size: int = 1, margin: int = 1, window: int = 3):
        self.n_splits = n_splits
        self.test_size = test_size
        self.margin = margin
        self.window = window

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y: Optional[Any] = None, groups: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        unique_dates = sorted(groups.unique())
        rank_dates = {date:rank for rank, date in enumerate(unique_dates)}
        X['index_time'] = groups.map(rank_dates)
        X = X.reset_index(drop = True)
        index_time_list = list(rank_dates.values())

        for i in reversed(range(1, self.n_splits + 1)):
            left_train = int((index_time_list[-1] - i*self.test_size + 1 - self.window - self.margin)*(self.window/np.max([1,self.window])))
            right_train = index_time_list[-1] - i*self.test_size - self.margin + 1
            left_test = index_time_list[-1] - i*self.test_size + 1
            right_test = index_time_list[-1] - (i-1)*self.test_size + 1
            index_test = X.index.get_indexer(X.index[X.index_time.isin(index_time_list[left_test: right_test])])
            index_train = X.index.get_indexer(X.index[X.index_time.isin(index_time_list[left_train: right_train])])
            yield index_train, index_test

class Kraken:
    def __init__(self, estimator: BaseEstimator, cv: BaseCrossValidator, metric: Callable, meta_info_name: str):
        self.estimator = estimator
        self.cv = cv
        self.metric = metric
        self.meta_info_name = meta_info_name

    def get_rank_dict(self, X: np.ndarray, y: np.ndarray, list_of_vars: List[str], group_dt: Optional[np.ndarray]):
        self.dict_fold_importances = {'Feature': list_of_vars, 'abs_shap': np.zeros(len(list_of_vars))}
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, groups = group_dt), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            self.estimator.fit(X_train[list_of_vars], y_train.values)
            explainer = shap.Explainer(self.estimator)
            shap_values = explainer.shap_values(X_test[list_of_vars])
            self.dict_fold_importances['abs_shap'] += np.abs(shap_values).mean(axis=0)
        self.fe_dict = {key: value for key, value in zip(self.dict_fold_importances['Feature'], self.dict_fold_importances['abs_shap'])}
        self.rank_dict = {key: rank for rank, key in enumerate(sorted(self.fe_dict, key=self.fe_dict.get, reverse=True), 1)}

    def get_cross_val_score(self, X: np.ndarray, y: np.ndarray, var: str, old_scores: np.ndarray, selected_vars: Optional[List[str]] = None, group_dt: Optional[np.ndarray] = None, round_num: int = 3):
        if selected_vars is None:
            selected_vars = []
        selected_vars.append(var)
        list_scores = []
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, groups=group_dt), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            self.estimator.fit(X_train[selected_vars], y_train)
            error = round(self.metric(np.exp(y_test), np.exp(self.estimator.predict(X_test[selected_vars]))), round_num)
            list_scores.append(error)
        fold_scores = np.array(list_scores)
        summa = sum(fold_scores - old_scores < 0) * 1 + sum(fold_scores - old_scores > 0) * -1
        mean_cv_score = round(np.mean(fold_scores), round_num)
        return fold_scores, summa, mean_cv_score

    def get_vars(self, X: np.ndarray, y: np.ndarray, early_stopping_rounds: int = 30, summa_approve: int = 1, best_mean_cv: int = 100, vars_in_model: Optional[List] = list(), group_dt: Optional[np.ndarray] = None, round_num: int = 3, old_scores: Optional[np.ndarray] = None):
        self.round_num = round_num
        if old_scores == None:
            old_scores = np.array([100 for i in range(self.cv.get_n_splits())])
        iteration_step = 0
        the_list_from_which_we_take_vars = [i for i in list(self.rank_dict.keys()) if i not in vars_in_model]
        feature_was_added = True
        while feature_was_added:
            iteration_step = 0
            var_for_add = ''
            print('начинаем след этап', best_mean_cv)
            best_positive_groups = summa_approve
            for var in the_list_from_which_we_take_vars:
                iteration_step += 1
                if iteration_step > early_stopping_rounds:
                    print(f'early_stopping_rounds {early_stopping_rounds}')
                    break
                fold_scores, summa, mean_cv_score = self.get_cross_val_score(X = X, y = y, var = var, old_scores = old_scores, selected_vars = vars_in_model.copy(), group_dt = group_dt, round_num = self.round_num)
                if (summa > best_positive_groups) or (summa == best_positive_groups and mean_cv_score < best_mean_cv):
                    best_positive_groups = summa
                    best_mean_cv = mean_cv_score
                    old_scores = fold_scores
                    var_for_add = var
                    iteration_step = 0
                    print(f'new var_for_add ! {var_for_add}')
            if var_for_add != '':
                vars_in_model.append(var_for_add)
                the_list_from_which_we_take_vars.remove(var_for_add)
                print('едем дальше')
                print('в итоге получили список', vars_in_model)
                list_meta = ['vars_list'] + [best_positive_groups] + [best_mean_cv] + old_scores.tolist()
                df_meta = pd.DataFrame(list_meta).T
                df_meta.columns = ['vars', 'summa', 'mean_cv_scores'] + ['cv' + str(i) for i in range(1, self.cv.get_n_splits() + 1)]
                df_meta.at[0, 'vars'] = vars_in_model.copy()
                try:
                    df_meta_info = pd.concat([df_meta_info, df_meta])
                except:
                    df_meta_info = df_meta.copy()
                df_meta_info.to_csv(f'df_meta_info_{self.meta_info_name}.csv')
                continue
            else:
                feature_was_added = False
        print('мы сошлись')
        print(vars_in_model)
        print(best_mean_cv)
        return vars_in_model

class AddTrain:
    """
    Class for add train
    """
    def __init__(self, df_: pd.DataFrame, model_path: 'str', train_end: str, oot_dates: List[str], vsp_test: np.array, used_features: List[str]):
        """
        Initialize AddTrain class with given df_, model, train_end, oot_dates.
        Args:
            df_ (pd.DataFrame): dataset with all features and target
            model_path (str): old model path from oper plan
            train_end (str): last report date in train set from develop process
            oot_dates (str): list oot dates in df_ from develop process
            vsp_test (np.array): set of test(oos) urf_code_map
            used_features (List): old model has incorrect naming features, that`s why need write explicit
        """
        self.df_ = df_
        self.model_path = model_path
        self.train_end = train_end
        self.oot_dates = oot_dates
        self.vsp_test = vsp_test
        self.used_features = used_features

    def scoring_constant_model(self):
        """
        Scoring old model. Method creates self.results_scor_constant with metrics
        """
        with open(self.model_path, 'rb') as mod_pkl:
            model = pickle.load(mod_pkl)
        cond1_oot = (self.df_['dt'] > self.train_end)
        X_oot = self.df_[cond1_oot]
        y_oot = np.log(X_oot['target'])
        print('*-*-*-*-*-*-*-*-*-*- oot *-*-*-*-*-*-*-*-*-*-')
        print(X_oot['dt'].value_counts().sort_index())

        macro_list = []
        for dt, subset in X_oot.groupby('dt'):
            y_pred_oot = np.exp(model.predict(subset[self.used_features]))
            mape_oot = round(mean_absolute_percentage_error(subset['target'], y_pred_oot), 2)
            macro_oot = round(y_pred_oot.sum(), 2)
            macro_fact = subset['target'].sum()
            ape_macro = round(100*(macro_oot - macro_fact)/macro_fact, 2)

            macro_list.append([dt, mape_oot, macro_oot, macro_fact, ape_macro])
            
        self.results_scor_constant = pd.DataFrame(macro_list, columns = ['dt', 'const_mape_oot', 'const_macro_oot', 'const_macro_fact', 'const_ape_macro'])
        self.results_scor_constant['const_mape_oot'] = self.results_scor_constant['const_mape_oot'] * (-1)

    def scoring_update_model(self, window: int, n_splits: int, test_size: int, margin: int, lgbm_params: dict, early_stopping_rounds: int, round_num: int, metric: Callable):
        """
        Method creates new model for every report date and emulates scoring with add train.
        """
        with open(self.model_path, 'rb') as mod_pkl:
            old_model = pickle.load(mod_pkl)
        macro_list = []
        for i, _ in enumerate(sorted(self.df_[self.df_['dt'] > self.train_end]['dt'].unique()[0:-3]), 1):
            if i == 1:
                cond1_train = (self.df_['dt'] <= pd.to_datetime(self.train_end) + MonthEnd(len(self.oot_dates)))
                cond2_train = (~self.df_['urf_code_map'].isin(self.vsp_test))
                X_train = self.df_[cond1_train & cond2_train]
                y_train = np.log(X_train['target'])
                cond1_test = (self.df_['urf_code_map'].isin(self.vsp_test))
                X_test = self.df_[cond1_train & cond1_test]
                y_test = np.log(X_test['target'])
                cond1_oot = (self.df_['dt'] == pd.to_datetime(self.train_end) + MonthEnd(len(self.oot_dates) + 2))
                X_oot = self.df_[cond1_oot]
                y_oot = np.log(X_oot['target'])
            elif i > 1:
                cond1_train = (self.df_['dt'] <= pd.to_datetime(self.train_end) + MonthEnd(len(self.oot_dates) + i - 1))
                cond2_train = (~self.df_['urf_code_map'].isin(self.vsp_test))
                X_train = self.df_[cond1_train & cond2_train]
                y_train = np.log(X_train['target'])
                cond1_test = (self.df_['urf_code_map'].isin(self.vsp_test))
                X_test = self.df_[cond1_train & cond1_test]
                y_test = np.log(X_test['target'])
                cond1_oot = (self.df_['dt'] == pd.to_datetime(self.train_end) + MonthEnd(len(self.oot_dates) + i + 1))
                X_oot = self.df_[cond1_oot]
                y_oot = np.log(X_oot['target'])
            print('*-*-*-*-*-*-*-*-*-*- start split train/test/oot *-*-*-*-*-*-*-*-*-*-')
            print(f'step_{i}')
            train_test_vc = pd.merge(X_train['dt'].value_counts().sort_index().reset_index(), X_test['dt'].value_counts().sort_index().reset_index(), how = 'outer', on = 'index')
            stats_val_cnt = pd.merge(train_test_vc, X_oot['dt'].value_counts().sort_index().reset_index(), how = 'outer', on = 'index')
            stats_val_cnt.columns = ['dt', 'cnt_train', 'cnt_oos', 'cnt_oot']
            display(stats_val_cnt)
            old_kwargs = {"bagging_fraction":old_model.bagging_fraction, "lambda_l1":old_model.lambda_l1, "learning_rate":old_model.learning_rate, "max_bin":old_model.max_bin, "max_depth":old_model.max_depth, "n_estimators":old_model.n_estimators, "num_leaves":old_model.num_leaves, "objective":old_model.objective, "random_state":old_model.random_state, "verbosity":old_model.verbosity}   
            model = LGBMRegressor(**old_kwargs)
            model.fit(X_train[self.used_features], y_train)
            dt = X_oot['dt'].unique()[0]
            y_pred_oot = np.exp(model.predict(X_oot[self.used_features]))
            mape_oot = round(mean_absolute_percentage_error(X_oot['target'], y_pred_oot), 2)
            macro_oot = round(y_pred_oot.sum(), 2)
            macro_fact = X_oot['target'].sum()
            ape_macro = round(100*(macro_oot - macro_fact)/macro_fact, 2)
            macro_list.append([dt, mape_oot, macro_oot, macro_fact, ape_macro])
        self.results_scor_update = pd.DataFrame(macro_list, columns = ['dt', 'update_w_mape_oot', 'update_w_macro_oot', 'update_w_macro_fact', 'update_w_ape_macro'])
        self.results_scor_update['update_w_mape_oot'] = self.results_scor_update['update_w_mape_oot'] * (-1)

    def final_report(self):
        report = pd.merge(self.results_scor_constant, self.results_scor_update, how = 'left', on = 'dt')
        report['diff'] = round(100*(report['update_w_mape_oot'] - report['const_mape_oot'])/report['const_mape_oot'], 2)
        return report
    
    def scoring_update_model(self, start_date: str, window: int, n_splits: int, test_size: int, margin: int, lgbm_params: dict, early_stopping_rounds: int, round_num: int, metric: Callable):
        """
        Метод создает новую модель для каждой даты отчета и эмулирует скоринг с добавлением обучения.

        Параметры:
        window (int): размер окна для DateTimeSeriesSplit.
        n_splits (int): количество разбиений в DateTimeSeriesSplit.
        test_size (int): размер тестовой выборки в DateTimeSeriesSplit.
        margin (int): маржа между тренировочным и тестовым набором в DateTimeSeriesSplit.
        lgbm_params (dict): параметры для инициализации LGBMRegressor.
        early_stopping_rounds (int): количество раундов для ранней остановки в Kraken.
        round_num (int): количество знаков после запятой для округления результатов.
        metric (Callable): метрика для оценки модели (например, mean_absolute_percentage_error).
        """
            # Загрузка старой модели
        with open(self.model_path, 'rb') as file:
            old_model = pickle.load(file)

        start_month_dt = pd.to_datetime(start_month)
        results = []
        meta_info = []

        print(f"Начинаем обработку данных, начиная с {start_month_dt.strftime('%Y-%m')}")

        while start_month_dt <= self.df_['dt'].max():
            print(f"Обрабатываем месяц {start_month_dt.strftime('%Y-%m')}")

            # Разделение на train и OOT
            train_data = self.df_[self.df_['dt'] < start_month_dt]
            oot_data = self.df_[self.df_['dt'] >= start_month_dt]

            # Инициализация DateTimeSeriesSplit и Kraken
            cv_datetime = DateTimeSeriesSplit(window=window, n_splits=n_splits, test_size=test_size, margin=margin) 
            group_dt = train_data['dt']
            model = LGBMRegressor(**lgbm_params)  # Необходимо инициализировать с параметрами
            selector = Kraken(model, cv_datetime, metric, 'updated_model')  # Необходимо инициализировать с параметрами

            # Подбор фичей на основе SHAP значений
            selector.get_rank_dict(train_data, np.log(train_data['target']), self.used_features, group_dt=train_data['dt'])
            new_vars_class = selector.get_vars(train_data, np.log(train_data['target']), vars_in_model=[], early_stopping_rounds=early_stopping_rounds, group_dt=train_data['dt'], round_num=round_num)

            # Обучение новой модели с отобранными переменными
            model.fit(train_data[new_vars_class], np.log(train_data['target']))

            # Оценка новой модели на OOT данных
            y_pred_new = np.exp(model.predict(oot_data[new_vars_class]))
            mape_new = mean_absolute_percentage_error(oot_data['target'], y_pred_new)

            # Оценка старой модели на OOT данных
            y_pred_old = np.exp(old_model.predict(oot_data[self.used_features]))
            mape_old = mean_absolute_percentage_error(oot_data['target'], y_pred_old)

            # Сравнение старой и новой модели
            if mape_new < mape_old:
                print(f"Новая модель ({mape_new}) лучше старой ({mape_old}) для {start_month_dt.strftime('%Y-%m')}")
                old_model = model
                # Сохраняем новую модель
                with open(self.model_path, 'wb') as file:
                    pickle.dump(model, file)
                results.append({'month': start_month_dt.strftime('%Y-%m'), 'model': 'new', 'mape': mape_new})
            else:
                print(f"Старая модель ({mape_old}) лучше новой ({mape_new}) для {start_month_dt.strftime('%Y-%m')}")
                results.append({'month': start_month_dt.strftime('%Y-%m'), 'model': 'old', 'mape': mape_old})

            # Сохраняем метаинформацию
            meta_info.append({
                'month': start_month_dt.strftime('%Y-%m'),
                'features': new_vars_class,
                'mape_new': mape_new,
                'mape_old': mape_old
            })

            start_month_dt += pd.DateOffset(months=1)

        # Сохранение метаинформации в CSV
        meta_info_df = pd.DataFrame(meta_info)
        meta_info_df.to_csv('meta_info.csv', index=False)

        return pd.DataFrame(results)
