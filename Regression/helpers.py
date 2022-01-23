import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import cross_validate


# ---------------------------------------------------------------------------------------------
def plot_numerical_attributes(df_full: pd.DataFrame, target: str, numeric_attr: list):
    """Main purpose to detect correlation and outliers"""
    # 4x3 blocks
    block_size = 12
    blocks = [numeric_attr[n:n + block_size] for n in range(0, len(numeric_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            sns.regplot(x=column, y=target, data=df_full, ax=ax,
                        order=3, ci=None, line_kws={'color': 'black'})
            ax.tick_params(labelrotation=45)
            plt.tight_layout()
        fig.suptitle(f'Numeric columns to {target}', fontsize=10, y=1.0)
    plt.show()


# ---------------------------------------------------------------------------------------------
def histplot_numerical_attributes(df_full: pd.DataFrame, numeric_attr: list):
    """Main purpose to detect skewed features"""
    # 4x3 blocks
    block_size = 12
    blocks = [numeric_attr[n:n + block_size] for n in range(0, len(numeric_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            sns.histplot(x=column, ax=ax, data=df_full)
            ax.tick_params(labelrotation=45)
            plt.tight_layout()
        fig.suptitle(f'Numeric columns', fontsize=10, y=1.0)
    plt.show()


# ---------------------------------------------------------------------------------------------
def plot_categorical_attributes(df_full: pd.DataFrame, target: str, categorical_attr: list):
    """Main purpose to detect target correlation"""
    # 4x3 blocks
    block_size = 12
    blocks = [categorical_attr[n:n + block_size] for n in range(0, len(categorical_attr), block_size)]

    for block in blocks:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        for column, ax in zip(block, axes):
            # Sort by median
            sort_d = df_full.groupby(column)[target].median().sort_values(ascending=False)
            sns.boxplot(x=column, y=target, data=df_full, ax=ax,
                        palette='plasma', order=sort_d.index)
            ax.tick_params(labelrotation=45)
            plt.tight_layout()
        fig.suptitle(f'Categorical columns to {target}', fontsize=10, y=1.0)
    plt.show()


# ---------------------------------------------------------------------------------------------
def plot_correlations(df_final: pd.DataFrame, target: str):
    correlations = df_final.corrwith(df_final[target]).iloc[:-1].to_frame()
    correlations['Abs Corr'] = correlations[0].abs()
    sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(sorted_correlations.to_frame()[sorted_correlations >= 0.5], cmap='coolwarm',
                annot=True, vmin=-1, vmax=1, ax=ax)
    plt.show()


# ---------------------------------------------------------------------------------------------
def plot_dist3(df_final: pd.DataFrame, target: str):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # Histogram
    ax[0].set_title('Histogram')
    sns.histplot(df_final.loc[:, target], kde=True, ax=ax[0])

    # Probability Plot
    ax[1].set_title('Probability Plot')
    stats.probplot(df_final.loc[:, target].fillna(np.mean(df_final.loc[:, target])), plot=ax[1])
    ax[1].get_lines()[0].set_markerfacecolor('#e74c3c')
    ax[1].get_lines()[0].set_markersize(12.0)

    # Box Plot
    ax[2].set_title('Box Plot')
    sns.boxplot(df_final.loc[:, target], ax=ax[2])

    plt.suptitle(target, fontsize=24)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
def model_check(x: pd.DataFrame, y_: pd.DataFrame, models: list, models_names: list, cv):
    model_table = pd.DataFrame()
    index = 0
    for model, name in zip(models, models_names):
        model_table.loc[index, 'Name'] = name
        cv_results = cross_validate(model, x, y_, cv=cv, scoring='neg_root_mean_squared_error',
                                    return_train_score=True, n_jobs=-1)
        model_table.loc[index, 'Train RMSE'] = -cv_results['train_score'].mean()
        model_table.loc[index, 'Test RMSE'] = -cv_results['test_score'].mean()
        model_table.loc[index, 'Test Std'] = cv_results['test_score'].std()

        index += 1

    return model_table.sort_values(by=['Test RMSE'], ascending=True)


# ---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
