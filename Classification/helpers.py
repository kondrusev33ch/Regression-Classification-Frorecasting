import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------
def numerical_correlations(df_full: pd.DataFrame):
    correlation = df_full.corr()

    sns.set(font_scale=0.7)
    mask = np.triu(correlation.corr())
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, mask=mask, linewidths=1, cbar=False)
    plt.show()


# ---------------------------------------------------------------------------------------------
def plot_numerical_features(df_full: pd.DataFrame, target: str, numeric_attr: list,
                            on_x: int = 3, on_y: int = 2):
    block_size = on_x * on_y
    blocks = [numeric_attr[n:n + block_size] for n in range(0, len(numeric_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(on_y, on_x, figsize=(on_x * 5, on_y * 5))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            sns.boxplot(x=target, y=column, data=df_full, ax=ax)
            plt.tight_layout()
        fig.suptitle(f'Numeric columns to {target}', fontsize=10, y=1.0)
    plt.show()


# ---------------------------------------------------------------------------------------------
def histplot_numerical_features(df_full: pd.DataFrame, numeric_attr: list,
                                on_x: int = 2, on_y: int = 2):
    block_size = on_x * on_y
    blocks = [numeric_attr[n:n + block_size] for n in range(0, len(numeric_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(on_y, on_x, figsize=(on_x * 5, on_y * 4))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            sns.histplot(x=column, ax=ax, data=df_full)
            ax.tick_params(labelrotation=45)
            plt.tight_layout()
        fig.suptitle(f'Numeric columns', fontsize=10, y=1.0)
    plt.show()


# --------------------------------------------------------------------------------------------
def plot_categorical_features(df_full: pd.DataFrame, target: str, categorical_attr: list,
                              on_x: int = 3, on_y: int = 2):
    block_size = on_x * on_y
    blocks = [categorical_attr[n:n + block_size] for n in range(0, len(categorical_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(on_y, on_x, figsize=(on_x * 5, on_y * 4))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            sns.histplot(x=column, hue=target, data=df_full, ax=ax, multiple="stack")
            plt.tight_layout()
        fig.suptitle(f'Categorical columns to {target}', fontsize=10, y=1.0)
    plt.show()
