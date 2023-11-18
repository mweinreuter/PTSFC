import pandas as pd
from datetime import timedelta


def most_recent_thursday(df):
    today = df.index.max()

    # Calculate days to the most recent Thursday
    days_to_thursday = (today.weekday() - 3) % 7
    recent_thursday = today - timedelta(days=days_to_thursday)

    return recent_thursday


def split_time(df, num_years=0, num_months=0, num_weeks=0, num_days=0, num_hours=0):

    # date_to_keep = df.index.max() - num_years - num_months - num_days
    split_date = df.index.max() - pd.DateOffset(years=num_years, months=num_months, weeks=num_weeks, days=num_days,
                                                hours=num_hours)
    df_b = df.loc[df.index <= split_date]
    df_a = df.loc[df.index > split_date]

    return df_b, df_a
