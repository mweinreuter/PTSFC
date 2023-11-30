from datetime import timedelta
import pandas as pd


def most_recent_thursday(df):
    today = df.index.max()

    # Calculate days to the most recent Thursday
    days_to_thursday = (today.weekday() - 3) % 7
    recent_thursday = today - \
        timedelta(days=days_to_thursday,
                  hours=today.hour, minutes=today.minute)

    # set time to 12:00 am
    most_recent_thursday = recent_thursday.replace(
        hour=12, minute=0, second=0, microsecond=0)

    return most_recent_thursday


def split_time(df, num_years=0, num_months=0, num_weeks=0, num_days=0, num_hours=0):

    # date_to_keep = df.index.max() - num_years - num_months - num_days
    split_date = df.index.max() - pd.DateOffset(years=num_years, months=num_months, weeks=num_weeks, days=num_days,
                                                hours=num_hours)
    df_b = df.loc[df.index <= split_date]
    df_a = df.loc[df.index > split_date]

    return df_b, df_a


def most_recent_wednesday(df, wednesday_morning=False):

    today = df.index.max()

    # Calculate days to the most recent Thursday
    days_to_wednesday = (today.weekday() - 2) % 7
    recent_wednesday = today - \
        timedelta(days=days_to_wednesday,
                  hours=today.hour, minutes=today.minute)

    if wednesday_morning == True:
        recent_wednesday = today - \
            timedelta(days=7,
                      hours=today.hour, minutes=today.minute)

    # set time to 12:00 am
    most_recent_wednesday = recent_wednesday.replace(
        hour=12, minute=0, second=0, microsecond=0)

    return most_recent_wednesday


def next_working_days(start_date, num_days=5):
    """
    Get the next N working days excluding weekends.

    Parameters:
    - start_date: The starting date in 'YYYY-MM-DD' format.
    - num_days: Number of working days to retrieve (default is 5).

    Returns:
    - A list of the next N working days.
    """
    start_date = pd.to_datetime(start_date) + pd.Timedelta(days=1)
    working_days = []

    while len(working_days) < num_days:

        # Check if the day is a weekday (Monday to Friday)
        if start_date.weekday() < 5:
            working_days.append(start_date.strftime('%Y-%m-%d'))

        # Increment the date by one day
        start_date += pd.Timedelta(days=1)

    return working_days
