# outstanding data points identified by outlier analysis

def delete_outstanding_dp(energy_df):

    indices = ['2023-03-26 02:00:00', '2022-03-27 02:00:00',
               '2022-10-30 02:00:00', '2023-10-29 02:00:00']

    indices_to_drop = [
        index for index in indices if index in energy_df.index]

    energy_df = energy_df.drop(indices_to_drop)

    return (energy_df)


def impute_outstanding_dp(energy_df):

    indices = ['2023-03-26 02:00:00', '2022-03-27 02:00:00',
               '2022-10-30 02:00:00', '2023-10-29 02:00:00']

    # impute with mean of energy_consumption between 2:00 am and 3:00 am
    for index in indices:
        if index in energy_df.index:
            energy_df.loc[index, 'energy_consumption'] = 42.880657

    return (energy_df)
