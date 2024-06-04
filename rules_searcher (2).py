import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import category_encoders as ce

from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import _tree
from pandas.io.excel import ExcelWriter



def filling(df):
    cat_vars = df.select_dtypes(include=[object]).columns
    num_vars = df.select_dtypes(include=[np.number]).columns
    df[cat_vars] = df[cat_vars].fillna('_MISSING_')
    df[num_vars] = df[num_vars].fillna(np.nan)
    return df

def replace_not_frequent_2(df, cols, num_min=100, value_to_replace = "_ELSE_", ignore_vars=[]):
    else_df = pd.DataFrame(columns=['var', 'list'])
    for i in cols:
        if i in ignore_vars:
            continue
        if i != 'date_requested' and i != 'credit_id':
            t = df[i].value_counts()
            q = list(t[t.values < num_min].index)
            if q:
                else_df = else_df.append(pd.DataFrame([[i, q]], columns=['var', 'list']))
            df.loc[df[i].value_counts(dropna=False)[df[i]].values < num_min, i] = value_to_replace
    else_df = else_df.set_index('var')
    return df, else_df

def drop_single_value_column(df, except_cols=[]):
    except_cols = set(except_cols)
    df2 = df.copy()
    for i in df2.columns:
        if i in except_cols:
            continue
        if df2[i].value_counts(dropna=False).shape[0]==1:
            df2.drop(i, axis=1, inplace=True)
    return df2  

# Делаем преобразования для категориальных признаков
def fit_category_encoding(df: pd.DataFrame, features: list, encoding_type: str='target_encoding',
                         target_name: str='target') -> dict:
    """
    Составляем таблицы соответсвия при кодировани для всех категориальных признаков,
    чтобы можно было кодировать и декодировать значения.
    Args:
        features: List[str] набор категориальных признаков, над которыми тербуется произвести преобразование
        encoding_type: способ кодирования, 'one_hot', 'target_encoding'

    Returns:
        cat_encoding: Dict[dict] словарь, где для каждой переменной задан свой словарь с соответствием

    """
    # TODO: подумать как сделать One-hot и мб какой-то ещё другой кодировщик
    cat_encoding = {}
    for feat in features:
        target_encoder = ce.target_encoder.TargetEncoder(smoothing=0.1)
        target_encoder.fit(df[[feat]], df[target_name])

        values = df[[feat]].drop_duplicates().copy()
        encoding = target_encoder.transform(values)
        values, encoding = values.values.reshape(1, -1)[0], encoding.values.reshape(1, -1)[0]
        cat_encoding[feat] = {values[i]: encoding[i] for i, _ in enumerate(values)}

    # Отдель добавить обработку новых категорий, будем выдавать среднее значение таргета по сэмплу.
    if '_ELSE_' not in set(cat_encoding.keys()):
        cat_encoding['_ELSE_'] = df[target_name].mean()

    return cat_encoding 

def transform_category_encoding(df: pd.DataFrame, features: list, cat_encoding: dict) -> pd.DataFrame:
    new_df = df.copy()
    for feat in features:
        encoding = cat_encoding[feat]
        # Заменяем все новые категории, на _ELSE_.
        categories = new_df[feat].unique()
        for cat in categories:
            if cat not in set(encoding.keys()):
                new_df[feat] = new_df[feat].replace(cat, '_ELSE_')

        new_df[feat] = new_df[feat].map(encoding)

    return new_df


def fit_nan_encoding(df: pd.DataFrame, features: list, nan_imputer: str='median', nan_custom: dict=None,
                 fill_value: int=None, iv_df: pd.DataFrame=None) -> dict:
    '''
    Формируем словарь с значениями для заполнения пропусков для каждого из признака.
    Args:
        nan_imputer: тип для заполнения, 'median', 'max_freq', 'constant', 'auto_iv'
        nan_custom: словарь для заполнения переменных заданным значением

    Return:
        nan_encoding: словарь <назвение переменной>: <значение для пропуска>

    '''
    if nan_custom is None:
        nan_encoding = {}
    else:
        nan_encoding = nan_custom
    features = [feat for feat in features if feat not in nan_encoding.keys()]

    if nan_imputer == 'median':
        for i, feat in enumerate(features):
            value = df[feat].median()
            nan_encoding[feat] = value

    elif nan_imputer == 'max_freq':
        for i, feat in enumerate(features):
            # Считаем индекс (значение переменной), которое чаще всего встречается.
            value = df[feat].value_counts().sort_values(ascending=False).index[0]
            nan_encoding[feat] = value

    elif nan_imputer == 'constant':
        if fill_value is None:
            raise Exception("ERROR, you put nan_imputer='constant', but forget to specify fill_value, check fill_value parameter")

        for i, feat in enumerate(features):
            nan_encoding[feat] = fill_value

    elif nan_imputer == 'auto_iv':
        if iv_df is None:
            raise ValueError("ERROR, you put nan_imputer='auto_iv', but forget to specify iv_df, check this parameter")

        for i, feat in enumerate(features):
            
            d3 = iv_df[iv_df['VAR_NAME'] == feat]
            dr_nan = d3.loc[d3['MIN_VALUE'].isna() == True, 'DR'].values[0]

            ind_for_nan = np.argmin(np.abs(d3[d3['MIN_VALUE'].isna() != True]['DR'] - dr_nan))
            segment_for_nan = d3[d3['MIN_VALUE'].isna() != True].iloc[ind_for_nan]

            value = df[(df[feat] > segment_for_nan['MIN_VALUE']) & (df[feat] < segment_for_nan['MAX_VALUE'] + 0.001)][feat].median()
            nan_encoding[feat] = value

    else:
        raise ValueError('ERROR, nan_imputer parameter is not correct, check it')

    return nan_encoding


def transform_nan_encoding(df: pd.DataFrame, features: list, nan_encoding: dict) -> pd.DataFrame:
    new_df = df.copy()
    for feat in features:
        value = nan_encoding[feat]
        new_df[feat].fillna(value, inplace=True)

    return new_df


def fit_inf_encoding(df: pd.DataFrame, num_feats: list) -> dict:
    '''
    
    Формируем словарь для заполнения бесконечностей(для каждого знака отдельно) в признаке.
    Бесконечности заменяются на максимальное(в случае положительной бесконечноти)
    или минимальное(в случае отрицательной) значение признака без них.
    
    '''

    inf_encoding = {}
    # Для положительных бесконечностей один словарь внутри итогового
    inf_encoding['plus_inf'] = {}
    for col in num_feats:
        val = df[df[col]!=np.inf][col].max()
        inf_encoding['plus_inf'][col] = val

    # Для отрицательных бесконечностей один словарь внутри итогового
    inf_encoding['minus_inf'] = {}
    for col in num_feats:
        val = df[df[col]!=-np.inf][col].min()
        inf_encoding['minus_inf'][col] = val

    return inf_encoding

def transform_inf_encoding(df: pd.DataFrame, num_feats: list, inf_encoding: dict) -> pd.DataFrame:

    #Заполняем значния с бесконечностями
    for col in num_feats:
        df[col] = df[col].replace({np.inf: inf_encoding['plus_inf'][col]})
        df[col] = df[col].replace({-np.inf: inf_encoding['minus_inf'][col]})

    return df


def tree_to_code_interv(tree, feature_names, df, y, BR_interval=[0, 0.05]):

    """Функция рекурсивно проходится по заданному дереву, и в случае обнаружения ветки
    с пониженным BR, записывает её путь и показатели(BR, segment_share) в список. Теперь в списке лежат все ветки одного дерева с пониженным BR,
    далее, я с помощью itertools.combinations(здесь можно юзать, так как в одном дереве максимум ветки 3 с пониженным BR), перебираю
    все возможные вариации объединения этих веток в единое правило. Так-же все объединения проверяю на BR, и вывожу значение с самым больших Segm_share
    BR_interval - указывать в долях"""

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature]
    
    result = list()

    def recurse(node, depth, s):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 3)

            sample_count = tree_.n_node_samples[node]
            class_count = tree_.value[node][0]
            segment_share = round(sample_count / len(df), 3)
            bad_rate = round(class_count[1] / (class_count[0] + class_count[1]), 3)

            if BR_interval[0] <= bad_rate <= BR_interval[1]:
                result.append((s[1:], round(bad_rate, 3), round(segment_share, 3)))
                return

            s1 = "({} <= {})".format(name, threshold)
            recurse(tree_.children_left[node], depth + 1, s + '&' + s1)
            s2 = "({} > {})".format(name, threshold)
            recurse(tree_.children_right[node], depth + 1, s + '&' + s2)
        else:
            sample_count = tree_.n_node_samples[node]
            class_count = tree_.value[node][0]
            segment_share = round(sample_count / len(df), 3)
            bad_rate = round(class_count[1] / (class_count[0] + class_count[1]), 3)
            if BR_interval[0] <= bad_rate <= BR_interval[1]:
                result.append((s[1:], round(bad_rate, 3), round(segment_share, 3)))
            return

    recurse(0, 1, '')
    
    return result


def rules_search_interv(X_train: pd.DataFrame, y_train: pd.Series, vars: list, BR_interval: list=[0, 0.05]) -> pd.DataFrame:

    """Строит множество деревьев решений на рандомно извлечённых признаках из vars,
    затем проходится по построенным деревьям, и в случае обнарежения ветки с нужным
    BR (попадающим в указанный интервал), записывает данную ветку в итог.
    Если в одном дереве присутстует несколько веток с нужным BR, 
    то функция их объединяет (по условиям BR_dwn - т.е. получаются 'деревья') и записывает в итог.
    
    Функция возвращает датафрэйм с указанным рулом, BR-ом после его применения и долей остатка выборки
    BR_interval - указывать в долях"""
    
    from sklearn.utils import shuffle
    result_rules = list()
    columns = vars

    for k in range(1000):
        columns = shuffle(columns, random_state=k)
        for i in range(0, len(columns), 8):
            curr = columns[i:i+7]
            clf_tree = DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=40
            )
            clf_tree.fit(X_train[curr], y_train)
            vals = tree_to_code_interv(clf_tree, curr, X_train, y_train, BR_interval)
    
            result_rules += vals
            
    return pd.DataFrame(data=result_rules, columns=['rule', 'BR', 'Segm_share']).sort_values('Segm_share', ascending=False) 


def check_rules(rules: list, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                             X_test: pd.DataFrame, y_test: pd.DataFrame, 
                             X_out: pd.DataFrame=None, y_out: pd.DataFrame=None) -> pd.DataFrame:

    """Функция просто рассчитывает характеристику рула на всех трёх выборках: BR, отсекаемый сегмент, и то, на сколько BR упал"""

    br_test = list()
    br_train = list()
    segm_test = list()
    segm_train = list()
    br_dwn_test = list()
    br_dwn_train = list()
    X_train['target'] = y_train
    X_test['target'] = y_test

    if X_out is not None:
        br_out = list()
        segm_out = list()
        br_dwn_out = list()
        X_out['target'] = y_out

    df = pd.DataFrame()

    for rl in rules:
        br_train.append(round(X_train.query(rl).target.mean(), 3))
        segm_train.append(round(X_train.query(rl).shape[0] /X_train.shape[0], 3))  
        br_dwn_train.append(round(X_train.query(rl).target.mean() / X_train.target.mean(), 3))   
        br_test.append(round(X_test.query(rl).target.mean(), 3))
        segm_test.append(round(X_test.query(rl).shape[0] /X_test.shape[0], 3))
        br_dwn_test.append(round(X_test.query(rl).target.mean() / X_test.target.mean(), 3))
        
        if X_out is not None:
            br_dwn_out.append(round(X_out.query(rl).target.mean() / X_out.target.mean(), 3))
            segm_out.append(round(X_out.query(rl).shape[0] /X_out.shape[0], 3))
            br_out.append(round(X_out.query(rl).target.mean(), 3))
    
    df['rule'] = rules
    df['br_train'] = br_train
    df['segm_train'] = segm_train
    df['br_dwn_train'] = br_dwn_train
    df['br_test'] = br_test
    df['segm_test'] = segm_test
    df['br_dwn_test'] = br_dwn_test

    if X_out is not None:
        df['br_out'] = br_out
        df['segm_out'] = segm_out
        df['br_dwn_out'] = br_dwn_out

    X_train.drop(columns=['target'], axis=1, inplace=True)
    X_test.drop(columns=['target'], axis=1, inplace=True)

    if X_out is not None:
        X_out.drop(columns=['target'], axis=1, inplace=True)   

    return df


def filter_df(nums: list, filters_list: list) -> str:
    
    """Функция принимает:
                nums - индексы фильтров (номер фильтра минус 1)
                filters_list - общий список фильтров, идущий на фильтрацию

       Предназначена для объединения фильтров для датарфэма
       по определённым индексам.
       (используется в последующих расчётных функциях)"""
    

    filters = filters_list.copy()

    res_filt = '(' + filters[nums[0]] + ')'
    if len(nums) == 1:
        return res_filt
    for i in range(1, len(nums)):
        res_filt += '|' + '(' + filters[nums[i]] + ')'

    #Возвращает:
    #       res_filt - объединённые фильтры в виде строки
    return res_filt


def calculate_values(df: pd.DataFrame, filters: str):

    """Фукнция принимает: 
                df - фильтруемый датафрэйм (уже с колонкой таргета)
                filters - строка с комбинацией фильтров"""

    BR_dwn = round((1 - df.query(filters, engine='python').target.mean()/df.target.mean())*100, 3) #снижение BR в %
    prt = round((df.query(filters, engine='python').shape[0] / df.shape[0])*100, 3) #часть оставшейся выборки
    BR_curr = round(df.query(filters, engine='python').target.mean()*100, 3) #значение BR
    #Возваращает значения BR_dwn, prt, BR_curr для отфильтрованного датафрэйма 
    return BR_dwn, prt, BR_curr


def random_step_by_step_learn_interv(X_train: pd.DataFrame, y_train: pd.Series, filters_list: list, BR_interval: list=[0, 0.05]):

    """Функция принимает:
            X_train - обучающий датафрэйм с необходимыми для фильтрации признаками
            y_train - соответствующий начальные df, со значением таргета(необходим для расчёта значений)
            filters_list - список используемых фильтров
            BR_interval - алгоритм будет пытаться подбирать такие рулы, чтобы BR результирующей выборки попадал в указанный интервал
            BR_interval - указывать в долях
                
            Данная функция для каждого фильтра, несколько раз добавляет рандомную последовательность фильтров.
        На каждом шаге добавления нового фильтра из рандомной последовательности, проводится проверка по trsh_hld,
        если результат удовлетворяет требованиям, то текущая комбинация добавляется в итоговый датарэйм,
        а фильтр добавляется в результирующий и процесс подбора продолжается.
        
            Таким образом, получается, что для каждого фильтра в итоговом датафрэйме, будет присутствовать
        несколько комбинаций с другими фильтрами, различной длины"""

    from sklearn.utils import shuffle

    X_train['target'] = y_train

    filter_ind = list(range(len(filters_list)))

    combinations = pd.DataFrame(columns=['filt_ind', 'BR_dwn_train', 'prt_train', 'BR_curr_train'])

    p = 1
    i = 0
    for ind in range(len(filters_list)):

        for _ in range(len(filters_list)):
            res = list()
            res.append(ind)

            train_filt = filter_df([ind], filters_list)

            BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, train_filt)
            curr_max_prt = prt_train

            filter_ind = shuffle(filter_ind, random_state=p+100)
            p += 1
            for j in filter_ind:
                if j in res:
                    continue
                else:
                    check = res + [j]
                    train_filt = filter_df(check, filters_list)

                    BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, train_filt)

                    if BR_interval[0] * 100 < BR_curr_train < BR_interval[1] * 100 and prt_train >= 1.02 * curr_max_prt:
                        res.append(j)
                        curr_max_prt = prt_train
                        combinations.loc[i, 'filt_ind'] = res.copy()
                        combinations.loc[i, 'BR_dwn_train'] = BR_dwn_train
                        combinations.loc[i, 'prt_train'] = prt_train
                        combinations.loc[i, 'BR_curr_train'] = BR_curr_train
                        i += 1

    combinations['filt_ind'] = combinations['filt_ind'].apply(sorted).apply(str)
    X_train.drop(columns=['target'], axis=1, inplace=True)
    #Возвращает датафрэйм со столбцом, с номерами используемых фильтров, 
    # и значениями показателей на всех выборках
    return combinations.drop_duplicates()


def Greed_search_learn_interv(X_train: pd.DataFrame, y_train: pd.Series, 
                              filters_list: list, BR_interval: list=[0, 0.05], max_res_cnt_filters: int=5) -> pd.DataFrame:

    """Функция принимает:
            X_train - обучающий датафрэйм с необходимыми для фильтрации признаками
            y_train - соответствующий начальные df, со значением таргета(необходим для расчёта значений)
            filters_list - список используемых фильтров
            BR_interval - алгоритм будет пытаться подбирать такие рулы, чтобы BR результирующей выборки попадал в указанный интервал
            max_res_cnt_filters - максимальное количество фильтров, которое можно добавить в процессе жадного алгоритма
                
            Данная функция для каждого фильтра, проходится по всему списку фильтров жадным поиском, и добавляет тот фильтр,
        с которым текущая комбинация фильтров даёт наилучший результат, при чём, каждый фильтр проверяется по требованиям trsh_hld,
        и специальному условию, что на каждом шаге, объём результирующей выборки должен расти. Если условия не выполняются,
        то для текущего фильтра поиск прекращается, результаты заносятся в итоговую таблицу, и начинается поиск для следующего фильтра.
        
        В результате, получается несколько (не больше количества фильтров) комбинаций, представленных в датафрэйме"""

    def rnd(x):
        return round(x)

    X_train['target'] = y_train

    filter_ind = list(range(len(filters_list)))

    combinations = pd.DataFrame(columns=['filt_ind', 'BR_dwn_train', 'prt_train', 'BR_curr_train'])

    i = 0
    for ind in filter_ind:
        res = list()
        res.append(ind)

        train_filt = filter_df(res, filters_list)

        BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, train_filt)

        curr_prt_max = round(prt_train)

        if BR_interval[0] * 100 < BR_curr_train < BR_interval[1] * 100:

            cnt = max_res_cnt_filters - 1
            while cnt != 0:
                curr = pd.DataFrame(columns=['filt_ind', 'BR_curr_train', 'prt_train'])

                j = 0
                for filt in filter_ind:

                    if filt not in res:
                        check = res + [filt]
                        train_filt = filter_df(check, filters_list)

                        BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, train_filt)
                        
                        if BR_interval[0] * 100 < BR_curr_train < BR_interval[1] * 100:
                            curr.loc[j, 'filt_ind'] = [filt]
                            curr.loc[j, 'BR_curr_train'] = BR_curr_train
                            curr.loc[j, 'prt_train'] = prt_train
                            j += 1

                if j != 0:

                    curr['BR_curr_train'] = curr['BR_curr_train'].apply(rnd)
                    curr['prt_train'] = curr['prt_train'].apply(rnd)

                    curr = curr.sort_values(['prt_train', 'BR_curr_train'], ascending=[False, False])
                    
                    if BR_interval[0] * 100 < curr['BR_curr_train'].values[0] < BR_interval[1] * 100 and curr['prt_train'].values[0] >= curr_prt_max:
                        res.append(curr['filt_ind'].values[0][0])
                        curr_prt_max = curr['prt_train'].values[0]
                    else:
                        break

                cnt -= 1
            
            train_filt = filter_df(res, filters_list)

            BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, train_filt)
                        
            combinations.loc[i, 'filt_ind'] = res.copy()
            combinations.loc[i, 'BR_dwn_train'] = BR_dwn_train
            combinations.loc[i, 'prt_train'] = prt_train
            combinations.loc[i, 'BR_curr_train'] = BR_curr_train
            i += 1
    combinations['filt_ind'] = combinations['filt_ind'].apply(sorted).apply(str)

    X_train.drop(columns=['target'], axis=1, inplace=True)   
    #Возвращает датафрэйм со столбцом, с номерами используемых фильтров, 
    # и значениями показателей на всех выборках
    return combinations.drop_duplicates()


def all_sample_data(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series, 
                    filters_list: list, combinations: pd.DataFrame,
                    X_out: pd.DataFrame=None, y_out: pd.Series=None) -> pd.DataFrame:

    """Функция принимает:
            X_train, X_test, X_out - датафрэймы с необходимыми для фильтрации признаками
            y_train, y_test, y_out - соответствующий df, со значением таргета(необходим для расчёта значений)
            filters_list - список используемых фильтров
            combinations - датафрэйм, являющийся результатом работы функций "Greed_search_learn" или "Random_step_by_step_learn"

                
            Данная функция по значениям комбинация в колонке 'filt_ind' датафрэйма combinations,
        рассчитывает значения 'BR_dwn', 'prt', 'BR_curr', для train, test и out выборок.
            
            В результате получается датафрэйм, с результатами для всех выборок по фильтрам,
        которые были получены, в процессе 'обучения' на train
        
        Данная функция возвращает начения в процентах, не в долях
        
        
        """

    #В filtr_ind - хранятся строки со списками, поэтому перевожуперевожу их обратно в списки
    to_check = list(combinations['filt_ind'].values)
    to_check = list(map(lambda x: x[1:-1].split(', '), to_check))

    #Для получения индексов, перевожу строковые числа в int

    if X_out is not None:
        result =  pd.DataFrame(columns=['filters', 'filt_cnt', 'BR_dwn_train', 'prt_train', 'BR_curr_train',
                                        'BR_dwn_test', 'prt_test', 'BR_curr_test', 
                                        'BR_dwn_out', 'prt_out', 'BR_curr_out'])
    if X_out is None:
        result =  pd.DataFrame(columns=['filters', 'filt_cnt', 'BR_dwn_train', 'prt_train', 'BR_curr_train',
                                        'BR_dwn_test', 'prt_test', 'BR_curr_test'])

    
    X_train['target'] = y_train
    X_test['target'] = y_test
    if X_out is not None:
        X_out['target'] = y_out
    
    i = 0
    for val in to_check:
        inxs = list(map(lambda x: int(x), val))

        filt = filter_df(inxs, filters_list)

        BR_dwn_train, prt_train, BR_curr_train = calculate_values(X_train, filt)
        BR_dwn_test, prt_test, BR_curr_test = calculate_values(X_test, filt)
        if X_out is not None:
            BR_dwn_out, prt_out, BR_curr_out = calculate_values(X_out, filt)
                    
        result.loc[i, 'filters'] = filt
        result.loc[i, 'filt_cnt'] = len(inxs)
        result.loc[i, 'BR_dwn_train'] = BR_dwn_train
        result.loc[i, 'prt_train'] = prt_train
        result.loc[i, 'BR_curr_train'] = BR_curr_train
        result.loc[i, 'BR_dwn_test'] = BR_dwn_test
        result.loc[i, 'prt_test'] = prt_test
        result.loc[i, 'BR_curr_test'] = BR_curr_test
        if X_out is not None:
            result.loc[i, 'BR_dwn_out'] = BR_dwn_out
            result.loc[i, 'prt_out'] = prt_out
            result.loc[i, 'BR_curr_out'] = BR_curr_out
        i += 1

    X_train.drop(columns=['target'], axis=1, inplace=True)
    X_test.drop(columns=['target'], axis=1, inplace=True)
    if X_out is not None:
        X_out.drop(columns=['target'], axis=1, inplace=True)

    return result


def rules_result(fin_rules, X_train, X_test, X_out, y_train, y_test, y_out):
    
    X_train['target'] = y_train
    X_test['target'] = y_test
    X_out['target'] = y_out

    X_all = pd.concat([X_train, X_test, X_out])

    # f_anoth = ''
    # for i in range(len(fin_rules)):
    #     f_anoth += '(~(' + fin_rules[i] + '))&'
    # f_anoth = f_anoth[:-1]

    print('Без фильтров: BR =', round(X_all.target.mean(), 3))
    print()

    for i in range(len(fin_rules)):
        print(f'{i+1} rule :  ', fin_rules[i])
    print()

    for i in range(len(fin_rules)):
        print(f'При применении {i+1}-ого рула:   Доля =', round(X_all.query(fin_rules[i]).shape[0]/X_all.shape[0], 4), 
                                           ', Изм.BR =', round(X_all.query(fin_rules[i]).target.mean()/X_all.target.mean(), 4), 
                                           ', Текущий BR =', round(X_all.query(fin_rules[i]).target.mean(), 3), 
                                           ', Доля от всего бэда', round(X_all.query(fin_rules[i]).target.sum() / X_all.target.sum(), 3))
    # print()

    # print('Сегмент не попавший ни в один рул:   Доля =', round(X_all.query(f_anoth).shape[0]/X_all.shape[0], 4), 
    #                                        ', Изм.BR =', round(X_all.query(f_anoth).target.mean()/X_all.target.mean(), 4), 
    #                                        ', Текущий BR =', round(X_all.query(f_anoth).target.mean(), 3), 
    #                                        ', Доля от всего бэда', round(X_all.query(f_anoth).target.sum() / X_all.target.sum(), 3))
    # print('Его рул - логическое отрицание всех рулов:', f_anoth)
    X_train.drop(columns=['target'], axis=1, inplace=True)
    X_test.drop(columns=['target'], axis=1, inplace=True)
    X_out.drop(columns=['target'], axis=1, inplace=True)
    return 


def rf_feature_selection(df: pd.DataFrame, y: pd.Series, top_n: int=20) -> pd.DataFrame:
    '''
    Считаем feature_importances с использование Случайного леса

    top_n: количество признаков для отбора

    '''
    model = RandomForestClassifier(random_state=142)
    model.fit(df, y)

    feature_imp = pd.DataFrame(model.feature_importances_, index=df.columns, columns=['feature_importance'])
    feature_imp = feature_imp.sort_values(by='feature_importance', ascending=False)

    return list(feature_imp[:top_n].index), feature_imp


def gini_month_selection(X: pd.DataFrame, df: pd.DataFrame, gini_min: float=0.05,
                         num_bad_intervals: int=2, target_name: str='target',
                        date_name: str='date_requested', intervals: str='month') -> tuple[list, pd.DataFrame]:
    '''
    Отбор признаков по однофакторной оценке gini по месяцам.
    Отбираем переменные, для которых для каждого месяца gini выше gini_min,
    допускается если gini ниже, но только если таких месяцев <= num_bad_moths.

    X: pd.DataFrame тренировочный набор преобразованных данных (X_train)
    df: pd.DataFrame тренировочный набор данных, содержащих date_requested и target (df_train)
    gini_min: минимальный порог gini для отбора
    num_bad_intervals: количество интервалов времени, в которых gini может быть меньше заданного
    target_name: имя таргета в df
    intervals: интервал времени, который берём - month или week

    Пример:
        gini_feats, df_gini_months = new_functions.gini_month_selection(X_train, df_train)

    '''
    df_x_month = pd.concat([X.reset_index(drop=True), df[[date_name, target_name]].reset_index(drop=True)], axis=1)
    if intervals == 'month':
        df_x_month['requested_month_year'] = df_x_month[date_name].apply(lambda x: str(x)[:7])
    elif intervals == 'week':
        df_x_month['requested_month_year'] = df_x_month[date_name].dt.strftime('%Y-%U')
    else:
        df_x_month['requested_month_year'] = df_x_month[date_name].apply(lambda x: str(x)[:7])
    vars_woe = X.columns

    requested_month_year = np.sort(df_x_month['requested_month_year'].unique())
    df_gini_months = pd.DataFrame(np.zeros((len(vars_woe), len(requested_month_year))), columns=requested_month_year)
    df_gini_months.index = vars_woe

    # Для каждого месяца и для каждой переменной рассчитываем однофакторный gini
    for month_year in requested_month_year:
        df_tmp = df_x_month[df_x_month['requested_month_year'] == month_year]

        for x in vars_woe:
            vars_t = [x] #vars_current + [x]
            df_train_m = df_tmp[vars_t]
            y_train = df_tmp[target_name]

            if y_train.value_counts().shape[0] < 2:
                # Таргет состоит только из одного класса
                Gini_train = -1
            else:
                _logreg = LogisticRegression().fit(df_train_m, y_train)

                predict_proba_train = _logreg.predict_proba(df_train_m)[:, 1]
                Gini_train = round(2 * roc_auc_score(y_train, predict_proba_train) - 1, 3)
            
            df_gini_months.loc[x, month_year] = Gini_train

    # Отбираем признаки, для которых количество плохо предсказанных месяцев меньше заданного числа.
    good_features = df_gini_months[((df_gini_months < gini_min).sum(axis=1) <= num_bad_intervals)].index

    df_gini_months.reset_index(inplace=True)
    df_gini_months = df_gini_months.rename(columns={'index': 'vars'})
    return good_features, df_gini_months