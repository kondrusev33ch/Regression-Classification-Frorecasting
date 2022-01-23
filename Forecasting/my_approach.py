"""
References:
https://www.kaggle.com/hiro5299834/store-sales-ridge-voting-bagging-et-bagging-rf
https://www.kaggle.com/andrej0marinchenko/hyperparamaters#DeterministicProcess
https://www.kaggle.com/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide
"""

import pandas as pd
import numpy as np
import plotly.express as px
from tabulate import tabulate
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf

import helpers as h
from custom_regressor import CustomRegressor

# ----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Target and general thoughts
    # ==========================================================================================
    # The first thing I do, is reading dataset notes, provided by author.
    # https://www.kaggle.com/c/store-sales-time-series-forecasting/data
    # We have 7 (actually 6) csv files to work with, each of them we should analyse individually.
    # Our target: predict sales for the thousands of product families sold at Favorita stores.
    #
    # Now let's get closer look at each table.

    # Train and Test datasets
    # ==========================================================================================
    train = pd.read_csv('data/train.csv', dtype={'store_nbr': 'category'},
                        usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'])
    test = pd.read_csv('data/test.csv', dtype={'store_nbr': 'category'},
                       usecols=['store_nbr', 'family', 'date', 'onpromotion'])

    h.df_statistics_with_date(train, 'Train')
    # Summary:
    #   [+] Missing values: 0
    #   [-] Missing dates: ['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25']

    h.df_statistics_with_date(test, 'Test')
    # Summary:
    #   [+] Missing values: 0
    #   [+] Missing dates: []

    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    # In our train and test datasets we have one interesting feature - onpromotion,
    # which means total number of items in a product family that were being promoted
    # at a giving date.
    # Sounds like it should influence well on sales.
    print(train.corr('spearman').sales.loc['onpromotion'])
    # 0.538021816358609
    # Not bad result, but how can we use it? I have one idea

    # Check store sales
    temp = train.set_index('date').groupby('store_nbr').resample('D').sales.sum().reset_index()
    fig = px.line(temp, x='date', y='sales', color='store_nbr',
                  title='Daily total sales of the stores')
    fig.show()

    # Some of the stores have sales only after 2014, 2015, or 2017
    # Let's fix it
    print(train.shape)  # => (3000888, 6)
    train = train[~((train.store_nbr == '52') & (train.date < "2017-04-20"))]
    train = train[~((train.store_nbr == '22') & (train.date < "2015-10-09"))]
    train = train[~((train.store_nbr == '42') & (train.date < "2015-08-21"))]
    train = train[~((train.store_nbr == '21') & (train.date < "2015-07-24"))]
    train = train[~((train.store_nbr == '29') & (train.date < "2015-03-20"))]
    train = train[~((train.store_nbr == '20') & (train.date < "2015-02-13"))]
    train = train[~((train.store_nbr == '53') & (train.date < "2014-05-29"))]
    train = train[~((train.store_nbr == '36') & (train.date < "2013-05-09"))]
    print(train.shape)  # => (2780316, 6)

    # Transactions dataset
    # ==========================================================================================
    transactions = pd.read_csv('data/transactions.csv', dtype={'store_nbr': 'category'})

    h.df_statistics_with_date(transactions, 'Transactions')
    # Summary:
    #   [+] Missing values: 0
    #   [-] Missing dates: ['2013-12-25', '2014-12-25', '2015-12-25', '2016-01-01',
    #                       '2016-01-03', '2016-12-25']

    transactions['date'] = pd.to_datetime(transactions['date'])

    # Proof that transactions are highly correlated with sales
    temp = pd.merge(train.groupby(['date', 'store_nbr']).sales.sum().reset_index(),
                    transactions, how='left')
    print(temp.corr('spearman').sales.loc['transactions'])
    # 0.8174644354591597

    # Now we can proof that stores on holidays make more money than on working days
    temp = transactions.copy()
    temp['year'] = temp.date.dt.year
    temp['day_of_week'] = temp.date.dt.dayofweek + 1
    temp = temp.groupby(['year', 'day_of_week']).transactions.mean().reset_index()
    fig = px.line(temp, x='day_of_week', y='transactions', color='year', title='Transactions')
    fig.show()

    # After the visual analysis, it is obvious that we should set holidays correctly to
    # get better results.
    # This is all for transactions, we do not need them for training models.

    # Holidays Events dataset
    # ==========================================================================================
    holidays = pd.read_csv('data/holidays_events.csv',
                           converters={'locale_name': h.strip_spaces})  # removes spaces

    h.df_statistics_with_date(holidays, 'Holidays')
    # Summary:
    #   [+] Missing values: 0
    #   [+] Missing dates: [...] - it is ok not to have holidays every day

    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays.set_index('date', inplace=True)

    # Let's do a closer look at columns components
    # By printing unique labels, we can check data on misspells, and get better data understanding
    print('Holidays types:', holidays['type'].unique())
    print('Holidays region types:', holidays['locale'].unique())
    print('Holidays locale names:', holidays['locale_name'].unique())
    # Summary:
    #   [+] No misspells
    # We already know about type and transferred columns from dataset description.
    # And now we understand what locale and locale_name are.
    # Now: we need to create a full calendar and specify working days and not working days.

    # Calendar
    holidays_rdy = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31'))
    holidays_rdy['day_of_week'] = holidays_rdy.index.dayofweek + 1  # Monday = 1, Sunday = 7
    holidays_rdy['work_day'] = True
    holidays_rdy.loc[holidays_rdy['day_of_week'] > 5, 'work_day'] = False  # for saturdays and sundays

    # Fixing index duplicates in holidays dataset
    duplicates = holidays[holidays.index.duplicated(keep=False)]
    print(duplicates['locale_name'])
    # date
    # 2012-06-25        Imbabura
    # 2012-06-25       Latacunga
    # 2012-06-25         Machala
    # 2012-07-03    SantoDomingo
    # 2012-07-03        ElCarmen
    #                   ...
    # 2017-07-03    SantoDomingo
    # 2017-12-08            Loja
    # 2017-12-08           Quito
    # 2017-12-22         Salinas
    # 2017-12-22         Ecuador

    # This was done manually
    duplicates = [('2012-06-25', 'Latacunga Machala'), ('2012-07-03', 'ElCarmen'),
                  ('2012-12-22', 'Ecuador'), ('2012-12-24', 'Ecuador'),
                  ('2012-12-31', 'Ecuador'), ('2013-05-12', 'Ecuador'),
                  ('2013-06-25', 'Machala Latacunga'), ('2013-07-03', 'SantoDomingo'),
                  ('2013-12-22', 'Salinas'), ('2014-06-25', 'Machala Imbabura Ecuador'),
                  ('2014-07-03', 'SantoDomingo'), ('2014-12-22', 'Ecuador'),
                  ('2014-12-26', 'Ecuador'), ('2015-06-25', 'Imbabura Latacunga'),
                  ('2015-07-03', 'SantoDomingo'), ('2015-12-22', 'Salinas'),
                  ('2016-04-21', 'Ecuador'), ('2016-05-01', 'Ecuador'),
                  ('2016-05-07', 'Ecuador'), ('2016-05-08', 'Ecuador'),
                  ('2016-05-12', 'Ecuador'), ('2016-06-25', 'Imbabura Latacunga'),
                  ('2016-07-03', 'SantoDomingo'), ('2016-07-24', 'Guayaquil'),
                  ('2016-11-12', 'Ecuador'), ('2016-12-22', 'Salinas'),
                  ('2017-04-14', 'Ecuador'), ('2017-06-25', 'Latacunga Machala'),
                  ('2017-07-03', 'SantoDomingo'), ('2017-12-08', 'Quito'),
                  ('2017-12-22', 'Ecuador')]
    # No holidays was transferred in duplicates

    holidays = holidays.groupby(holidays.index).first()  # we left only first, but we need others too
    for date, locale_name in duplicates:
        holidays.loc[date, 'locale_name'] = holidays.loc[date, 'locale_name'] + ' ' + locale_name

    # Apply holidays to calendar
    holidays_rdy = holidays_rdy.merge(holidays, how='left', left_index=True, right_index=True)

    # type column: 'Work Day'
    holidays_rdy.loc[holidays_rdy['type'] == 'Work Day', 'work_day'] = True

    # type column: 'Holiday', 'Transfer', 'Additional', 'Bridge'
    holidays_rdy.loc[(holidays_rdy['type'] == 'Holiday') &
                     (holidays_rdy['locale_name'].str.contains('Ecuador', na=False)),
                     'work_day'] = False
    holidays_rdy.loc[(holidays_rdy['type'] == 'Transfer') &
                     (holidays_rdy['locale_name'].str.contains('Ecuador', na=False)),
                     'work_day'] = False
    holidays_rdy.loc[(holidays_rdy['type'] == 'Additional') &
                     (holidays_rdy['locale_name'].str.contains('Ecuador', na=False)),
                     'work_day'] = False
    holidays_rdy.loc[(holidays_rdy['type'] == 'Bridge') &
                     (holidays_rdy['locale_name'].str.contains('Ecuador', na=False)),
                     'work_day'] = False

    holidays_rdy.drop(['locale'], axis=1, inplace=True)

    # transferred column
    holidays_rdy.loc[holidays_rdy['transferred'] == True, 'work_day'] = True

    # type column: 'Event'
    # There are multiple events in dataset: Mother's day, Footbal championship, Black Friday,
    # Cyber Monday, Manabi Earthquake (about a month long)
    # We should understand how does it affect our sales.

    # First let's look at events
    events = holidays_rdy[holidays_rdy['type'] == 'Event']
    print(tabulate(events.head(10), headers='keys'))
    #               day_of_week  work_day    type    locale_name    description                                 transf
    # ----------  -------------  ----------  ------  -------------  ------------------------------------------  ------
    # 2014-05-11              7  False       Event   Ecuador        Dia de la Madre                             False
    # 2014-06-12              4  True        Event   Ecuador        Inauguracion Mundial de futbol Brasil       False
    # 2014-06-15              7  False       Event   Ecuador        Mundial de futbol Brasil: Ecuador-Suiza     False
    # 2014-06-20              5  True        Event   Ecuador        Mundial de futbol Brasil: Ecuador-Honduras  False
    # 2014-06-28              6  False       Event   Ecuador        Mundial de futbol Brasil: Octavos de Final  False
    # 2014-06-29              7  False       Event   Ecuador        Mundial de futbol Brasil: Octavos de Final  False
    # 2014-06-30              1  True        Event   Ecuador        Mundial de futbol Brasil: Octavos de Final  False
    # 2014-07-01              2  True        Event   Ecuador        Mundial de futbol Brasil: Octavos de Final  False
    # 2014-07-04              5  True        Event   Ecuador        Mundial de futbol Brasil: Cuartos de Final  False
    # 2014-07-05              6  False       Event   Ecuador        Mundial de futbol Brasil: Cuartos de Final  False

    # All events are national, no events were transferred.
    # We should set one label for all football events, same for earthquake.

    # I do it for simplicity
    holidays_rdy.loc[holidays_rdy['description'].str.contains('Terremoto', na=False),
                     'description'] = 'Earthquake'
    holidays_rdy.loc[holidays_rdy['description'].str.contains('futbol', na=False),
                     'description'] = 'Football'
    events = holidays_rdy[holidays_rdy['type'] == 'Event']

    # Check for misspells
    print(events['description'].unique())
    # ['Dia de la Madre' 'Football' 'Black Friday' 'Cyber Monday' 'Earthquake']

    # Print mean sales
    sales = train.groupby(['date']).sales.sum()
    events = events.merge(sales, how='left', left_index=True, right_index=True)
    print(events.groupby(['description']).sales.mean())
    print('All sales mean:', sales.mean())
    # description
    # Black Friday       647508.781658
    # Cyber Monday       777344.484674
    # Dia de la Madre    632258.371941
    # Earthquake         866433.436092
    # Football           569432.001918
    # All sales mean: 637556.3849186868

    # Imprecise method because we do not have enough data, but Earthquake and Cyber Monday
    # definitely should be considered during training, + Black Friday. Sales are not depends
    # much on Football and Mother's day.

    # descriptions
    descriptions = pd.get_dummies(holidays_rdy['description'])[['Earthquake', 'Cyber Monday', 'Black Friday']]
    holidays_rdy = holidays_rdy.merge(descriptions, how='left', left_index=True, right_index=True)

    # Fill NaNs
    holidays_rdy['locale_name'].fillna('Ecuador', inplace=True)

    # Get rid of useless columns
    holidays_rdy.drop(['type', 'description', 'transferred'], axis=1, inplace=True)

    # If you want to merge two dataframes, they should have same indexes, later we will need it
    holidays_rdy['date'] = holidays_rdy.index
    holidays_rdy['date'] = pd.to_datetime(holidays_rdy['date'])
    holidays_rdy['date'] = holidays_rdy['date'].dt.to_period('D')
    holidays_rdy = holidays_rdy.set_index(['date'])

    holidays_rdy = pd.get_dummies(holidays_rdy, columns=['day_of_week'])

    print(tabulate(holidays_rdy.head(), headers='keys'))
    # date        work_day    locale_name      Earthquake    Cyber Monday    Black Friday    day_of_week_1  ...
    # ----------  ----------  -------------  ------------  --------------  --------------  ---------------  ...
    # 2013-01-01  False       Ecuador                   0               0               0                0  ...
    # 2013-01-02  True        Ecuador                   0               0               0                0  ...
    # 2013-01-03  True        Ecuador                   0               0               0                0  ...
    # 2013-01-04  True        Ecuador                   0               0               0                0  ...
    # 2013-01-05  True        Ecuador                   0               0               0                0  ...

    # Oil dataset
    # ==========================================================================================
    oil = pd.read_csv('data/oil.csv')

    h.df_statistics_with_date(oil, 'Oil')
    # Summary:
    #   [!] Missing values: 43
    #   [!] Missing dates: [...] -

    oil['date'] = pd.to_datetime(oil['date'])

    # Fix missing values and missing dates by interpolation
    # Resample
    oil = oil.set_index('date')['dcoilwtico'].resample(
        'D').sum().reset_index()  # add missing dates and fill NaNs with 0

    # Interpolate
    oil['dcoilwtico'] = np.where(oil['dcoilwtico'] == 0, np.nan, oil['dcoilwtico'])  # replace 0 with NaN
    oil['dcoilwtico_interpolated'] = oil.dcoilwtico.interpolate()  # fill NaN values using an interpolation method

    temp = oil.melt(id_vars=['date'], var_name='Legend')
    fig = px.line(temp.sort_values(['Legend', 'date'], ascending=[False, True]), x='date',
                  y='value', color='Legend', title='Daily Oil Price')
    fig.show()

    oil_rdy = oil.loc[:, ['date', 'dcoilwtico_interpolated']]
    oil_rdy.iloc[0, 1] = 93.1

    assert oil_rdy.isna().sum().sum() == 0

    oil_rdy['date'] = pd.to_datetime(oil_rdy['date'])
    oil_rdy['date'] = oil_rdy['date'].dt.to_period('D')
    oil_rdy = oil_rdy.set_index(['date'])

    # But what if oil prices don't influence sales? Why do we need an oil dataset?
    # Here you are.
    # For some columns we can see strong correlation.

    import matplotlib.pyplot as plt


    def plot_sales_and_oil_dependency():
        a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(),
                     oil.drop("dcoilwtico", axis=1), how="left")
        c = a.groupby("family").corr("spearman").reset_index()
        c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

        _, axes = plt.subplots(7, 5, figsize=(20, 20))
        for i, fam in enumerate(c.family):
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=axes[i // 5, i % 5])
            axes[i // 5, i % 5].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6],
                                          fontsize=12)
            axes[i // 5, i % 5].axvline(x=70, color='r', linestyle='--')

        plt.tight_layout(pad=5)
        plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize=20)
        plt.show()


    plot_sales_and_oil_dependency()

    # Add rolling mean and lags
    oil_rdy['rolling_mean_7'] = oil_rdy['dcoilwtico_interpolated'].rolling(7).mean()
    oil_rdy.fillna(93.1, inplace=True)

    _ = plot_pacf(oil_rdy.rolling_mean_7, lags=12, method='ywm')  # 2 lags?
    plt.show()

    for i in range(1, 3):
        oil_rdy[f'oil_lag_{i}'] = oil_rdy.rolling_mean_7.shift(i)
    oil_rdy.fillna(93.1, inplace=True)

    print(oil_rdy)
    #             dcoilwtico_interpolated  rolling_mean_7  oil_lag_1  oil_lag_2
    # date
    # 2013-01-01                93.100000       93.100000  93.100000  93.100000
    # 2013-01-02                93.140000       93.100000  93.100000  93.100000
    # 2013-01-03                92.970000       93.100000  93.100000  93.100000
    # 2013-01-04                93.120000       93.100000  93.100000  93.100000
    # 2013-01-05                93.146667       93.100000  93.100000  93.100000
    # ...                             ...             ...        ...        ...
    # 2017-08-27                46.816667       47.490000  47.629048  47.765714
    # 2017-08-28                46.400000       47.348571  47.490000  47.629048
    # 2017-08-29                46.460000       47.178571  47.348571  47.490000
    # 2017-08-30                45.960000       46.822857  47.178571  47.348571
    # 2017-08-31                47.260000       46.825714  46.822857  47.178571

    # Stores dataset
    # ==========================================================================================
    stores = pd.read_csv('data/stores.csv', index_col='store_nbr',
                         converters={'city': h.strip_spaces, 'state': h.strip_spaces})

    # Summary:
    #   [+] Missing values: 0

    # As before lets look at the unique labels
    print('Cities:\n', stores['city'].unique())
    print('States:\n', stores['state'].unique())
    print('Store types:\n', stores['type'].unique())  # no type information was provided in data description
    print('Clusters:\n', sorted(list(stores['cluster'].unique())))

    # We should connect stores and holidays by location, because holidays can be local or
    # regional (in one city, or in the whole state)
    stores_rdy = stores.loc[:, ['city', 'state']]
    print(stores_rdy)
    #                    city                       state
    # store_nbr
    # 1                 Quito                   Pichincha
    # 2                 Quito                   Pichincha
    # 3                 Quito                   Pichincha
    # 4                 Quito                   Pichincha
    # 5          SantoDomingo  SantoDomingodelosTsachilas
    # ...                 ...                         ...
    # 50               Ambato                  Tungurahua
    # 51            Guayaquil                      Guayas
    # 52                Manta                      Manabi
    # 53                Manta                      Manabi
    # 54             ElCarmen                      Manabi

    # Preprocessing
    # ==========================================================================================
    # I want to split training and testing datasets to train models for each store separately.
    # Because of the local holidays and onpromotion.
    # Yeah, maybe it's not the best idea, but lets try.

    # y
    train['date'] = train['date'].dt.to_period('D')
    train_rdy = train.set_index(['store_nbr', 'family', 'date']).sort_index()
    print(train_rdy)
    # store_nbr family     date
    # 1         AUTOMOTIVE 2013-01-01   0.000000            0
    #                      2013-01-02   2.000000            0
    #                      2013-01-03   3.000000            0
    #                      2013-01-04   3.000000            0
    #                      2013-01-05   5.000000            0
    # ...                                    ...          ...
    # 9         SEAFOOD    2017-08-11  23.831000            0
    #                      2017-08-12  16.859001            4
    #                      2017-08-13  20.000000            0
    #                      2017-08-14  17.000000            0
    #                      2017-08-15  16.000000            0

    test['date'] = test['date'].dt.to_period('D')
    test_rdy = test.set_index(['store_nbr', 'family', 'date']).sort_index()
    print(test_rdy)
    #                                  onpromotion
    # store_nbr family     date
    # 1         AUTOMOTIVE 2017-08-16            0
    #                      2017-08-17            0
    #                      2017-08-18            0
    #                      2017-08-19            0
    #                      2017-08-20            0
    # ...                                      ...
    # 9         SEAFOOD    2017-08-27            0
    #                      2017-08-28            0
    #                      2017-08-29            0
    #                      2017-08-30            0
    #                      2017-08-31            0

    start_date = '2017-03-25'  # Start and end of the training date
    end_date = '2017-08-15'

    y_arr = []
    y_onpromotion_arr = []
    y_test_onpromotion_arr = []
    for nbr in stores.index:
        y = train_rdy.loc[str(nbr), 'sales']
        y_arr.append(y.unstack(['family']).loc[start_date:end_date])

        y_onpromotion = train_rdy.loc[str(nbr), 'onpromotion']
        y_onpromotion = y_onpromotion.unstack(['family']).loc[start_date:end_date].sum(axis=1)
        y_onpromotion.name = 'onpromotion'
        y_onpromotion_arr.append(y_onpromotion)

        y_test_onpromotion = test_rdy.loc[str(nbr), 'onpromotion']
        y_test_onpromotion = y_test_onpromotion.unstack(['family']).sum(axis=1)
        y_test_onpromotion.name = 'onpromotion'
        y_test_onpromotion_arr.append(y_test_onpromotion)

    print(y_arr[0])
    # family      AUTOMOTIVE  BABY CARE  ...  SCHOOL AND OFFICE SUPPLIES  SEAFOOD
    # date                               ...
    # 2017-03-25         5.0        0.0  ...                         0.0   37.283
    # 2017-03-26         5.0        0.0  ...                         1.0   16.902
    # 2017-03-27         6.0        0.0  ...                         0.0   33.669
    # 2017-03-28         4.0        0.0  ...                         2.0   21.657
    # 2017-03-29         2.0        0.0  ...                         0.0   34.074
    # ...                ...        ...  ...                         ...      ...
    # 2017-08-11         1.0        0.0  ...                         0.0   19.424
    # 2017-08-12         6.0        0.0  ...                         0.0   20.150
    # 2017-08-13         1.0        0.0  ...                         0.0   11.378
    # 2017-08-14         1.0        0.0  ...                         0.0   14.129
    # 2017-08-15         4.0        0.0  ...                         0.0   22.487

    # X
    fourier = CalendarFourier(freq='W', order=4)
    X_arr = []
    X_test_arr = []
    store_index = 1
    for y, y_onpromotion, y_test_onpromotion \
            in zip(y_arr, y_onpromotion_arr, y_test_onpromotion_arr):
        dp = DeterministicProcess(index=y.index, constant=False, order=1,
                                  seasonal=False, additional_terms=[fourier], drop=True)

        X = dp.in_sample()
        X_test = dp.out_of_sample(steps=16)

        # On promotion
        X = X.merge(y_onpromotion, how='left', left_index=True, right_index=True)
        X_test = X_test.merge(y_test_onpromotion, how='left', left_index=True, right_index=True)

        # Holidays
        X = X.merge(holidays_rdy, how='left', left_index=True, right_index=True)
        X_test = X_test.merge(holidays_rdy, how='left', left_index=True, right_index=True)

        store_state = stores.loc[store_index, 'state']
        store_city = stores.loc[store_index, 'city']

        # Apply local holidays
        for j in X.index:
            if X.loc[j, 'locale_name'].find(store_state) != -1 or X.loc[j, 'locale_name'].find(store_city) != -1:
                X.loc[j, 'work_day'] = False

        for j in X_test.index:
            if X_test.loc[j, 'locale_name'].find(store_state) != -1 or X_test.loc[j, 'locale_name'].find(
                    store_city) != -1:
                X_test.loc[j, 'work_day'] = False

        X.drop(['locale_name'], axis=1, inplace=True)
        X_test.drop(['locale_name'], axis=1, inplace=True)

        # Oil
        X = X.merge(oil_rdy, how='left', left_index=True, right_index=True)
        X_test = X_test.merge(oil_rdy, how='left', left_index=True, right_index=True)

        X_arr.append(X)
        X_test_arr.append(X_test)

        store_index += 1

    print(tabulate(X_arr[0].head(10), headers='keys'))
    # date          trend    sin(1,freq=W-SUN)    cos(1,freq=W-SUN)    sin(2,freq=W-SUN)  ...    oil_lag_2
    # ----------  -------  -------------------  -------------------  -------------------  ...  -----------
    # 2017-03-25        1            -0.974928            -0.222521             0.433884  ...      47.6529
    # 2017-03-26        2            -0.781831             0.62349             -0.974928  ...      47.5043
    # 2017-03-27        3             0                    1                    0         ...      47.3686
    # 2017-03-28        4             0.781831             0.62349              0.974928  ...      47.2457
    # 2017-03-29        5             0.974928            -0.222521            -0.433884  ...      47.1357
    # 2017-03-30        6             0.433884            -0.900969            -0.781831  ...      47.3271
    # 2017-03-31        7            -0.433884            -0.900969             0.781831  ...      47.6386
    # 2017-04-01        8            -0.974928            -0.222521             0.433884  ...      48.11
    # 2017-04-02        9            -0.781831             0.62349             -0.974928  ...      48.5729
    # 2017-04-03       10             0                    1                    0         ...      49.0352

    # Modelling
    # ==========================================================================================
    # First of all, let's try Ridge Regressor and see at the results
    ridge = make_pipeline(RobustScaler(),
                          Ridge(alpha=31.0))

    train_errors = []
    validation_errors = []

    # Collect errors for each store
    for X, y in zip(X_arr, y_arr):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                          random_state=1, shuffle=False)
        model = ridge.fit(X_train, y_train)
        train_pred = pd.DataFrame(model.predict(X_train), index=X_train.index,
                                  columns=y_train.columns).clip(0.0)
        val_pred = pd.DataFrame(model.predict(X_val), index=X_val.index,
                                columns=y_val.columns).clip(0.0)

        y_train = y_train.stack(['family']).reset_index()
        y_train['pred'] = train_pred.stack(['family']).reset_index().loc[:, 0]

        y_val = y_val.stack(['family']).reset_index()
        y_val['pred'] = val_pred.stack(['family']).reset_index().loc[:, 0]

        train_errors.append(y_train.groupby('family').apply(
            lambda r: mean_squared_log_error(r.loc[:, 0], r['pred'])))
        validation_errors.append(y_val.groupby('family').apply(
            lambda r: mean_squared_log_error(r.loc[:, 0], r['pred'])))

    # Sum of mean squared log error from validation dataset
    print(sum(validation_errors).sort_values(ascending=False))
    # family
    # SCHOOL AND OFFICE SUPPLIES    94.694688
    # LINGERIE                      25.520165
    # GROCERY II                    22.059989
    # LADIESWEAR                    19.574730
    # ...                                 ...
    # BREAD/BAKERY                   1.897852
    # GROCERY I                      1.812620
    # DAIRY                          1.739158
    # PRODUCE                        1.533028
    # BOOKS                          0.875229

    # Here we can see that SCHOOL AND OFFICE SUPPLIES error is much higher than others.
    # We need to create custom regressor, and specify different models to deal with this problem.

    # I took already working well custom regressor, and made parameters tuning for Ridge and SVR.
    # Get fitted models
    models = []
    for X, y in zip(X_arr, y_arr):
        model = CustomRegressor()
        model.fit(X, y)
        models.append(model)

    # Get predictions
    results = []
    for X_test, model, y in zip(X_test_arr, models, y_arr):
        y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns).clip(0.0)
        results.append(y_pred.stack(['family']))

    # Submission
    # ==========================================================================================
    # To create submission we need concatenate all predictions in one dataframe.
    # Note: originally data was sorted by store_nbr as string, so it looked like this 1, 10, 11, 12, ...
    # To concatenate predictions correctly
    # Get correct dates for submission
    dates = ['2017-08-16', '2017-08-17', '2017-08-18', '2017-08-19', '2017-08-20', '2017-08-21',
             '2017-08-22', '2017-08-23', '2017-08-24', '2017-08-25', '2017-08-26', '2017-08-27',
             '2017-08-28', '2017-08-29', '2017-08-30', '2017-08-31']

    # Get correct order for submission
    order = list(range(1, len(results) + 1))
    str_map = map(str, order)
    correct_order_str = sorted(list(str_map))
    int_minus_one = lambda element: int(element) - 1
    correct_order_int = list(map(int_minus_one, correct_order_str))

    # Create and fill list with predictions in the correct order
    data = []
    for date in dates:
        for i in correct_order_int:
            data += results[i].loc[date].to_list()

    # Create dataframe from the list
    result = pd.DataFrame(data, columns=['sales'])

    # We can use sample_submission.csv to make submission
    submission = pd.read_csv('data/sample_submission.csv')
    submission['sales'] = result['sales']
    print(submission)
    #             id        sales
    # 0      3000888     3.773495
    # 1      3000889     0.000000
    # 2      3000890     3.360279
    # 3      3000891  2433.554876
    # 4      3000892     0.328275
    # ...        ...          ...
    # 28507  3029395   396.466419
    # 28508  3029396   105.630065
    # 28509  3029397  1429.841644
    # 28510  3029398    97.169900
    # 28511  3029399    14.956587

    # Save submission
    submission.to_csv('submissions/submission.csv', index=False)
