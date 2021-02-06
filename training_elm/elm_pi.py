import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from art.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import math
import sys
import argparse
import os
from elm_model import ExtremeLearningMachine
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", default=10, type=int)
parser.add_argument("-n", "--n_unit", default=600, type=int)
parser.add_argument("-f", "--file_name_path", default="file_name_path", type=str)
args = parser.parse_args()

threshold = args.threshold
n_unit = args.n_unit
file_name = args.file_name_path


def standard_trans(x_train, x_test):
    stdsc = StandardScaler()
    x_train_std = stdsc.fit_transform(x_train)
    # print(stdsc.mean_,stdsc.var_)
    x_test_std = stdsc.transform(x_test)
    return (
        x_train_std.astype(np.float64),
        x_test_std.astype(np.float64),
        (stdsc.mean_, stdsc.var_),
    )


def shuffle_data(x_data, y_data):
    n_data = np.shape(y_data)[0]
    shuffled_indices = np.arange(n_data)
    np.random.shuffle(shuffled_indices)
    x_data = x_data[shuffled_indices]
    y_data = y_data[shuffled_indices]
    return x_data, y_data


def extract_data_balanced(df):
    benign = df["label"] == 0
    benign[24127:] = False
    df = pd.concat([df.loc[benign, :], df.loc[df["label"] == 1, :]], axis=0)
    return df


def extract_data_unbalanced_more_malicious(df):
    # b:6050 m:19500
    benign = df["label"] == 0
    benign[6051:] = False
    malicious = df["label"] == 1
    malicious_index = np.where(malicious == True)
    malicious_index = malicious_index[0][:19500]
    ind = np.zeros(len(malicious), dtype=bool)
    ind[malicious_index] = True
    df = pd.concat([df.loc[benign, :], df.loc[ind, :]], axis=0)
    return df


def extract_data_unbalanced_more_benign(df):
    # m:19500 b:6050
    benign = df["label"] == 0
    benign[19500:] = False
    print(df.loc[benign, :].shape)
    malicious = df["label"] == 1
    malicious_index = np.where(malicious == True)
    malicious_index = malicious_index[0][:6050]
    print(len(malicious_index))
    ind = np.zeros(len(malicious), dtype=bool)
    ind[malicious_index] = True
    df = pd.concat([df.loc[benign, :], df.loc[ind, :]], axis=0)
    return df


def get_average_result(perm_list):
    result = pd.DataFrame(
        data=0,
        columns=["importances_mean", "importances_std"],
        index=list(columns_dic.keys()),
    )
    for pi in perm_list:
        result += pi
    result = result / len(perm_list)
    print(result)
    return result


def compute_permutaion_importance(perm_list, x_train_std, y_train, model, columns_dic):
    y_train_one_hot = np.argmax(y_train, axis=1)
    result = permutation_importance(
        model,
        x_train_std,
        y_train_one_hot,
        scoring="accuracy",
        n_repeats=10,
        n_jobs=-1,
        random_state=71,
    )
    print(result)
    perm_imp_df = pd.DataFrame(
        {
            "importances_mean": result["importances_mean"],
            "importances_std": result["importances_std"],
        },
        index=list(columns_dic.keys()),
    )
    perm_list.append(perm_imp_df)
    return perm_list


def feature_selection(file_name, threshold, columns):
    if os.path.isfile(file_name):
        perm_imp_df = pd.read_csv(file_name, index_col=0)
        return list(perm_imp_df.index)[:threshold]
    else:
        if threshold == 25:
            return columns
        else:
            raise ValueError("file not exists")


def output_perm_imp(perm_list, columns_dic, n_unit):
    result = get_average_result(perm_list)
    perm_imp_df = pd.DataFrame(
        {
            "importances_mean": result["importances_mean"],
            "importances_std": result["importances_std"],
        },
        index=list(columns_dic.keys()),
    )
    print(perm_imp_df)
    perm_imp_df = perm_imp_df.sort_values("importances_mean", ascending=False)
    perm_imp_df.to_csv("./data/perm_imp_nunit{}.csv".format(n_unit))


def Training(x_data, y_data, n_unit, threshold, save_mode, shi_work):
    """
    Training a model by the entire dataset and save the trained model and the parameters.
    """
    stdsc = StandardScaler()
    x_data = stdsc.fit_transform(x_data)
    y_data = to_categorical(y_data, 2).astype(np.float64)
    model = ExtremeLearningMachine(n_unit=n_unit)
    model.fit(X=x_data, y=y_data)
    if not shi_work and save_mode:
        model.save_weights("./data/elm_threshold{}_nunit{}".format(threshold, n_unit))
        mean, var = stdsc.mean_, stdsc.var_
        np.savez(
            "./data/param_threshold{}_nunit{}".format(threshold, n_unit),
            mean=mean,
            var=var,
        )
    elif shi_work and save_mode:
        model.save_weights("./data/elm_shi{}".format(n_unit))
        mean, var = stdsc.mean_, stdsc.var_
        np.savez(
            "./data/param_shi",
            mean=mean,
            var=var,
        )


# Define parameters
extraction_dataset_mode = (
    "normal"  # bengin than malicous("btm") or malicous than bengin ("mtb") or "normal"
)
shi_work = False
save_mode = True
pi_mode = False

####################################

# load data
df = pd.read_csv(file_name)
# Replace nan
df = df.replace(np.nan, 0)
# Select extraction dataset mode
if extraction_dataset_mode == "btm":
    df = extract_data_unbalanced_more_benign(df)
elif extraction_dataset_mode == "mtb":
    df = extract_data_unbalanced_more_malicious(df)
elif extraction_dataset_mode == "normal":
    df = extract_data_balanced(df)

columns_dic = {
    column: index for index, column in enumerate(df.drop("label", axis=1).columns)
}


if shi_work:
    features = [
        "length",
        "max_consecutive_chars",
        "entropy",
        "n_ip",
        "n_countries",
        "mean_TTL",
        "stdev_TTL",
        "life_time",
        "active_time",
    ]
else:
    features = feature_selection(
        "./data/perm_imp_nunit{}.csv".format(n_unit),
        threshold=threshold,
        columns=list(columns_dic.keys()),
    )

df = df.loc[:, features + ["label"]]
x_data = df.drop("label", axis=1).values
y_data = df["label"].values


FOLD_NUM = 5
fold_seed = 71
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
fold_iter = folds.split(x_data)
perm_list = []
# acc,precision,recall,f1 train
acc_train_total = []
precision_train_total = []
recall_train_total = []
f1_train_total = []
# acc,precision,recall,f1 test
acc_test_total = []
precision_test_total = []
recall_test_total = []
f1_test_total = []
eval_result = {}
for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):
    print(f"Fold times:{n_fold}")
    x_train, x_test = x_data[trn_idx], x_data[val_idx]
    y_train, y_test = y_data[trn_idx], y_data[val_idx]
    x_train_std, x_test_std, _ = standard_trans(x_train, x_test)
    y_train, y_test = (
        to_categorical(y_train, 2).astype(np.float64),
        to_categorical(y_test, 2).astype(np.float64),
    )
    # Training
    model = ExtremeLearningMachine(n_unit=n_unit, activation=None)
    model.fit(x_train_std, y_train)
    # Results
    y_train_pred = np.argmax(model.transform(x_train_std), axis=1)
    y_train_true = np.argmax(y_train, axis=1)
    y_test_pred = np.argmax(model.transform(x_test_std), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    # evaluate train
    acc_train_total.append(accuracy_score(y_train_true, y_train_pred))
    precision_train_total.append(precision_score(y_train_true, y_train_pred))
    recall_train_total.append(recall_score(y_train_true, y_train_pred))
    f1_train_total.append(f1_score(y_train_true, y_train_pred))
    # evaluate test
    acc_test_total.append(accuracy_score(y_test_true, y_test_pred))
    precision_test_total.append(precision_score(y_test_true, y_test_pred))
    recall_test_total.append(recall_score(y_test_true, y_test_pred))
    f1_test_total.append(f1_score(y_test_true, y_test_pred))
    # permutation importance
    if threshold == 25 and pi_mode:
        perm_list = compute_permutaion_importance(
            perm_list, x_train_std, y_train, model, columns_dic
        )

eval_result["train_accuracy"] = np.average(acc_train_total)
eval_result["train_precision"] = np.average(precision_train_total)
eval_result["train_recall"] = np.average(recall_train_total)
eval_result["train_f1"] = np.average(f1_train_total)

eval_result["test_accuracy"] = np.average(acc_test_total)
eval_result["test_precision"] = np.average(precision_test_total)
eval_result["test_recall"] = np.average(recall_test_total)
eval_result["test_f1"] = np.average(f1_test_total)

eval_df = pd.DataFrame.from_dict(eval_result, orient="index")
eval_df = eval_df.rename(columns={0: threshold})

# Output results
if shi_work and save_mode:
    eval_df.to_csv("./eval_result/eval_shi_nunit{}.csv".format(n_unit))
elif save_mode:
    eval_df.to_csv(
        "./eval_unbalanced_more_benign/eval_threshold{}_nunit{}.csv".format(
            threshold, n_unit
        )
    )
if threshold == 25 and pi_mode:
    output_perm_imp(perm_list, columns_dic, n_unit)

# Training(x_data, y_data, n_unit, threshold, save_mode=True, shi_work=shi_work)