
def get_population(energy_df):
    energy_df['year'] = energy_df.index.year

    # update 2024 continuously
    population_mapping = {
        2016: 82.350,
        2017: 82.792,
        2018: 83.019,
        2019: 83.167,
        2020: 83.155,
        2021: 83.237,
        2022: 84.359,
        2023: 84.581,
        2024: 84.600  # check
    }

    energy_df['population'] = energy_df['year'].map(population_mapping)
    energy_df = energy_df.drop(columns=['year'])
    return energy_df
