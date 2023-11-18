import pandas as pd

from help_functions.prepare_data import split_time
from help_functions.calculate_score import evaluate_horizon


def evaluate_different_daxmodels(models, df, last_x, years=False, months=False, weeks=False):

    # Check that exactly one of the boolean parameters is True
    if sum([years, months, weeks]) != 1:
        raise ValueError(
            "Exactly one of the boolean parameters (years, months, weeks) must be True.")

    years = int(years)
    months = int(months)
    weeks = int(weeks)

    for m in models:
        m['evaluation'] = evaluate_daxmodel(
            m, df, last_x, years, months, weeks)


def evaluate_daxmodel(model, df, last_x=100, years=False, months=False, weeks=False):
    '''
    model
        forecasting model (dict containing of name and function for 5 forecasts)
    df
        data frame containing energy data of last wednesday as last data point
    last_x
        number of iterations calculating score
    years, months, weeks 
        set time intervals for iterations
    '''

    df_before = df
    evaluation = pd.DataFrame()

    for w in range(2, last_x):  # range 1 ausprobieren
        df_before, df_after = split_time(
            df_before, num_years=years, num_months=months, num_weeks=weeks)  # set to weeks again
        pred = model['function'](df_before)

        obs = pd.DataFrame(columns=['LogRetLag1'])
        for index, row in pred.iterrows():
            if index in df.index:
                obs.loc[index] = df.loc[index]['LogRetLag1']

        merged_df = pd.merge(pred, obs, left_index=True, right_index=True)
        # Add scores to the merged_df
        for index, row in merged_df.iterrows():
            quantile_preds = row[[
                'q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
            observation = row['LogRetLag1']
            score = evaluate_horizon(quantile_preds, observation)
            merged_df.at[index, 'score'] = score

        evaluation = pd.concat([evaluation, merged_df])

    return evaluation
