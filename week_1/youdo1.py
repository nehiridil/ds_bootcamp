from sklearn.datasets import fetch_california_housing, make_regression
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math


def reg(x, y, group, p=0.3, verbose=False):
    beta = np.random.random(2)
    gamma = dict((k, np.random.random(2)) for k in range(6))

    if verbose:
        st.write(beta)
        st.write(gamma)
        st.write(x)

    alpha = 0.002
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _k, _x, _y in zip(group, x, y):
            y_pred = p * (beta[0] + beta[1] * _x) + (1 - p) * (gamma[_k][0] + gamma[_k][1] * _x)

            g_b0 = -2 * p * (_y - y_pred)
            g_b1 = -2 * p * ((_y - y_pred) * _x)

            # st.write(f"Gradient of beta0: {g_b0}")

            g_g0 = -2 * (1 - p) * (_y - y_pred)
            g_g1 = -2 * (1 - p) * ((_y - y_pred) * _x)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            gamma[_k][0] = gamma[_k][0] - alpha * g_g0
            gamma[_k][1] = gamma[_k][1] - alpha * g_g1

            err += (_y - y_pred) ** 2

        print(f"{it} - Beta: {beta}, Gamma: {gamma}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta, gamma


def ls_l1(x, y, lam, alpha=0.0001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        if beta[0] >= 0:
            g_b0 = -2 * (y - y_pred).sum() + lam
        else:
            g_b0 = -2 * (y - y_pred).sum() - lam

        if beta[1] >= 0:
            g_b1 = -2 * (x * (y - y_pred)).sum() + lam
        else:
            g_b1 = -2 * (x * (y - y_pred)).sum() - lam

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


def ls_l2(x, y, lam, alpha=0.0001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


def ls(x, y, alpha=0.001, verbose=False) -> np.ndarray:
    beta = np.random.random(2)
    if verbose:
        st.write(beta)

    print("starting sgd")
    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x
        print(f"y_pred: {y_pred} ")

        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


def ls_custom(x, y, theta = 5, alpha=0.000001, verbose=False) -> np.ndarray:
    beta = np.random.random(2)
    if verbose:
        st.write(beta)

    print("starting sgd")
    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x
        print(f"y_pred: {y_pred} ")


        g_b0 = (((y_pred - y) * np.exp(np.absolute(y_pred-y)+5)) / np.power((np.exp(y_pred-y) + np.exp(theta)),2) * np.absolute(y_pred-y)).sum()
        g_b1 = ((x * (y_pred-y) * np.exp(np.absolute(y_pred-y)+5)) / np.power((np.exp(np.absolute(y_pred-y)) + np.exp(theta)),2) * np.absolute(y_pred-y)).sum()

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)
        print(g_b1)
        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta

def make_reg(n: int = 100, verbose: bool = False):
    rng = np.random.RandomState(0)
    x, y = make_regression(n, 1, random_state=rng,
                           noise=st.slider("Noise", 1., 100., value=10., help="Perturbation around mean"))
    if st.checkbox("Outlier"):
        x = np.concatenate((x, np.array([[-2], [0], [2], [-2], [0], [2]])))
        y = np.concatenate((y, np.array([300, 300, 300, 300, 300, 300])))
    df = pd.DataFrame(dict(x=x[:, 0], y=y))
    if verbose:
        st.dataframe(df)

    if st.checkbox("Plotly OLS fit"):
        fig = px.scatter(df, x="x", y="y", trendline="ols")
    else:
        fig = px.scatter(df, x="x", y="y")
    st.plotly_chart(fig, use_container_width=True)
    return x, y

def convexity(verbose: bool = False):
    st.header("Let's Generate a Regression Dataset")
    n = st.slider("Number of Samples", 10, 1000, value=100, help="Number of samples generated")
    x, y = make_reg(n)
    m, n = 50, 50
    b0 = np.linspace(-25, 25, m)
    b1 = np.linspace(-25, 25., n)

    loss = np.empty((m, n))

    l = st.selectbox("Loss", ["Mean Square", "Mean Square + L2", "Mean Square + L1","Custom Loss Func"])

    if l == "Mean Square":
        beta = ls(x[:, 0], y)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = ((y - (_b1 * x + _b0)) ** 2).mean()
    elif l == "Mean Square + L2":
        lam = st.slider("Lambda", 0., 10., value=1.)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = np.sqrt(np.power((y - _b1 * x - _b0), 2).mean()) + lam * np.linalg.norm(
                    np.array([_b0, _b1]))
    elif l == "Mean Square + L1":
        lam = st.slider("Lambda", 0., 10., value=1.)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = np.sqrt(np.power((y - _b1 * x - _b0), 2).mean()) + lam * np.abs(np.array([_b0, _b1])).sum()

    elif l == "Custom Loss Func":
        m, n = 50, 50
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = (1 / (1 + np.exp(5+(np.absolute(y - (_b1 * x + _b0)) * -1)))).mean()

    # FIX: Axis naming

    fig =go.Figure(data=go.Contour(
        z=loss,
        x=b0,
        y=b1
    ))

    # fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### You Do 1")

#convexity()

st.markdown(r"""
        Our model has 2 parametes $\beta_0$ and $\beta_1$. Model can be defined as 
        """)
st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")

st.markdown(r"Our custom loss function can be written as")

st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{\frac{1}{(1 + e^{(-|-(y_i - (\beta_0 + \beta_1 x_i))|)+\theta})}}")

n = 100

cal_housing = fetch_california_housing()
x = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

df = pd.DataFrame(
    dict(MedInc=x['MedInc'], Price=cal_housing.target))

x = x['MedInc'].values

losses = []
y_hat_list = []
distances = []
threshold = st.slider("Threshold", 0., 10., value=5.)
for y_actual in np.linspace(-10, 10, 100):
    for y_pred in np.linspace(-10, 10, 100):
        distances.append(y_actual-y_pred)
        losses.append((1 / (1 + np.exp(threshold + (np.absolute(y_actual-y_pred) * -1)))).mean())

l = pd.DataFrame(dict(distances=distances, losses=losses))
st.markdown("#### Custom Loss Func. Convexity Check")

st.latex(r"distance_i = y_i - (\beta_0 + \beta_1 x_i)")

fig = px.scatter(l, x="distances", y="losses")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
            r"Given that $L$ is convex wrt both $\beta_0$ and $\beta_1$, "
            r"we can use Gradient Descent to find  $\beta_0^{*}$ and $\beta_1^{*}$ by using partial derivatives")
st.latex(
    r"\frac{\partial L}{\partial \beta_0} =  \sum^{N}_{i=1}{ \frac {(\beta_0 + \beta_1 * x_i - y_i) * e^{|\beta_0 + \beta_1 * x_i - y_i|+\theta}} "
    r"{(e^{\beta_0 + \beta_1 * x_i-y_i} + e^5)^2 *  |\beta_0 + \beta_1*x_i - y_i| }}")
st.latex(
    r"\frac{\partial L}{\partial \beta_1} =  \sum^{N}_{i=1}{ \frac {x_i*(\beta_0 + \beta_1*x_i -y_i) * e^{|\beta_0-y_i+x_i*\beta_1|+\theta}} "
    r"{(e^{|\beta_0-y_i+x_i*\beta_1|} + e^5)^2 * |\beta_0 - y_i + x_i*\beta_1|}}")

beta_custom = ls_custom(x, y, threshold)

st.latex(fr"\beta_0={beta_custom[0]:.4f}, \beta_1={beta_custom[1]:.4f}")
##WE DO
beta_ls = ls(x,y)



y_pred_custom = beta_custom[0] + beta_custom[1] * x
#y_pred_wedo_ls = beta_ls[0] + beta_ls[1] * x

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data points'))
fig.add_trace(go.Scatter(x=x, y=y_pred_custom, mode='lines', name='predictions_with_custom_loss_func'))
#fig.add_trace(go.Scatter(x=x, y=y_pred_wedo_ls, mode='lines', name='predictions_with_ls'))

st.plotly_chart(fig, use_container_width=True)


