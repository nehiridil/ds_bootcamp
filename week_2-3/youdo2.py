import pandas as pd
import numpy as np


def ls_recommender(r, alpha=0.001) -> np.ndarray:
    beta_user = np.random.random(len(r))
    beta_item = np.random.random(len(r[0]))
    not_nan_indices = np.argwhere(np.logical_not(np.isnan(r)))

    print("starting sgd")
    y_pred = np.ones(r.shape) * np.nan

    for iteration in range(1000):
        for index in not_nan_indices:
            y_pred[index[0]][index[1]] = beta_user[index[0]] + beta_item[index[1]]

        g_b0 = -1 * np.nansum(np.dstack((r, -y_pred)), 2)
        g_b1 = -1 * np.nansum(np.dstack((r, -y_pred)), 2)

        # print(f"({i}) beta_user: {beta_user}, beta_item: {beta_item}, gradient: {g_b0} {g_b1}")

        beta_prev_user = np.copy(beta_user)
        beta_prev_item = np.copy(beta_item)

        for i in range(len(beta_user)):
            beta_user[i] = beta_user[i] - (np.nansum(g_b0[i]) * alpha)

        for j in range(len(beta_item)):
            beta_item[j] = beta_item[j] - (np.nansum(g_b1[:, j]) * alpha)

        if np.linalg.norm(beta_user - beta_prev_user) < 0.0001 and np.linalg.norm(beta_item - beta_prev_item) < 0.0001:
            print(f"I do early stoping at iteration {iteration}")
            break

    return beta_user, beta_item


def ls_recommender_modified(r, alpha=0.001, hyperparameter_lambda=0.01) -> np.ndarray:
    beta_user = np.random.random(len(r))
    beta_item = np.random.random(len(r[0]))
    not_nan_indices = np.argwhere(np.logical_not(np.isnan(r)))

    print("starting sgd")
    y_pred = np.ones(r.shape) * np.nan

    for iteration in range(100):
        for index in not_nan_indices:
            y_pred[index[0]][index[1]] = beta_user[index[0]] + beta_item[index[1]]

        g_b_user = (-1 * np.nansum(np.dstack((r, -y_pred)), 2)) + (hyperparameter_lambda * np.nansum(beta_user))
        g_b_item = (-1 * np.nansum(np.dstack((r, -y_pred)), 2)) + (hyperparameter_lambda * np.nansum(beta_item))

        # print(f"({i}) beta_user: {beta_user}, beta_item: {beta_item}, gradient: {g_b0} {g_b1}")

        beta_prev_user = np.copy(beta_user)
        beta_prev_item = np.copy(beta_item)

        for i in range(len(beta_user)):
            beta_user[i] = beta_user[i] - (np.nansum(g_b_user[i]) * alpha)

        for j in range(len(beta_item)):
            beta_item[j] = beta_item[j] - (np.nansum(g_b_item[:, j]) * alpha)

        if np.linalg.norm(np.nansum(beta_user - beta_prev_user)) < 0.01 and np.linalg.norm(
                np.nansum(beta_item - beta_prev_item)) < 0.01:
            print(f"I do early stoping at iteration {iteration}")
            break

    return beta_user, beta_item

def calc_err(beta_user, beta_item, r):
  residual_sum_of_squares = 0
  not_nan_indices = np.argwhere(np.logical_not(np.isnan(r)))
  for index in not_nan_indices:
    i = index[0]
    j = index[1]
    y_hat = beta_user[i] + beta_item[j]
    y = r[i][j]
    residual_sum_of_squares += (y_hat - y) ** 2
  error = residual_sum_of_squares/2
  return error

def calc_err_modified(beta_user, beta_item, r, hyperparameter_lambda = 0.01):
  residual_sum_of_squares = 0
  not_nan_indices = np.argwhere(np.logical_not(np.isnan(r)))
  for index in not_nan_indices:
    i = index[0]
    j = index[1]
    y_hat = beta_user[i] + beta_item[j]
    y = r[i][j]
    residual_sum_of_squares += (y_hat - y) ** 2
    error_first_part = residual_sum_of_squares/2
    error_modified_part = (hyperparameter_lambda * (np.nansum(np.power(beta_user,2)) + np.nansum(np.power(beta_item,2)))) / 2
    error = error_first_part + error_modified_part
  return error



if __name__ == '__main__':

    df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                     names=['user_id', 'item_id', 'rating', 'timestamp'])

    r = df.pivot(index='user_id', columns='item_id', values='rating').values

    #PART 1

    beta_user, beta_item = ls_recommender(r, alpha=0.001)
    part1_err = calc_err(beta_user, beta_item, r)
    print("Part 1 Error: " + str(part1_err))

    #PART 2

    not_nan_indices = np.argwhere(np.logical_not(np.isnan(r)))
    idx = np.random.choice(np.arange(100_00), 100, replace=False)
    test_indices = not_nan_indices[idx]

    # train test split
    r_train = r.copy()
    r_test = r.copy()
    for test_index in test_indices:
        r_train[test_index[0]][test_index[1]] = np.nan

    for index in not_nan_indices:
        if index not in test_indices:
            r_test[index[0]][index[1]] = np.nan

    lambda_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    for hyperparameter_lambda in lambda_list:
        print("**For lambda " + str(hyperparameter_lambda))
        beta_user_modified, beta_item_modified = ls_recommender_modified(r_train, alpha=0.001,
                                                                         hyperparameter_lambda=hyperparameter_lambda)
        error = calc_err_modified(beta_user_modified, beta_item_modified, r_test, 0.01)
        print('Error: ' + str(error))

