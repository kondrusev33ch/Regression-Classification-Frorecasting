import pandas as pd
from tabulate import tabulate


# ----------------------------------------------------------------------------------------------
def df_statistics_with_date(df: pd.DataFrame, name: str):
    print(f'[{name}] ===========================================================================')
    print('Missing values:', df.isna().sum().sum())

    df_copy = df.copy()
    df_copy = df_copy.set_index('date')
    df_copy.index = pd.to_datetime(df_copy.index)
    print('Missing dates:', pd.date_range(start=df_copy.index[0],
                                          end=df_copy.index[-1]).difference(df_copy.index))

    print(tabulate(df.head(), headers='keys'))
    print(df.dtypes)
    print()


# ----------------------------------------------------------------------------------------------
def strip_spaces(a_str_with_spaces):
    return a_str_with_spaces.replace(' ', '')


# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
