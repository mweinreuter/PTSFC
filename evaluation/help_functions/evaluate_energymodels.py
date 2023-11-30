import pandas as pd

from help_functions.prepare_data import split_time
from help_functions.calculate_score import evaluate_horizon


def evaluate_different_energymodels(models, df, last_x=100, years=False, months=False, weeks=False):

    # Check that exactly one of the boolean parameters is True
    if sum([years, months, weeks]) != 1:
        raise ValueError(
            "Exactly one of the boolean parameters (years, months, weeks) must be True.")

    years = int(years)
    months = int(months)
    weeks = int(weeks)

    for m in models:
        print(
            f'*********** Start the evaluation of model {m["name"]}***********')
        m['evaluation'] = evaluate_energymodel(
            m, df, last_x, years, months, weeks)


# requiremend: no missing data
def evaluate_energymodel(model, df, last_x=100, years=False, months=False, weeks=False):
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

    for w in range(2, last_x):
        if w % 10 == 0:
            print(f'Iteration {w} of {last_x}')
        df_before, df_after = split_time(
            df_before, num_years=years, num_months=months, num_weeks=weeks)

        if callable(model['function']):
            pred = model['function'](
                df_before)
        else:
            evaluation = pd.DataFrame()
            break
        # pred = model['function'](df_before)
        obs = pd.DataFrame(                                                                                #
            {'energy_consumption': df.loc[pred['date_time']]['energy_consumption']})
        pred = pred.set_index('date_time')
        merged_df = pd.merge(pred, obs, left_index=True, right_index=True)

        # Add scores to the merged_df
        for index, row in merged_df.iterrows():
            quantile_preds = row[[
                'q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']]
            observation = row['energy_consumption']
            score = evaluate_horizon(quantile_preds, observation)
            merged_df.at[index, 'score'] = score

        print(pred.index)  # delete

        evaluation = pd.concat([evaluation, merged_df])

    return evaluation
