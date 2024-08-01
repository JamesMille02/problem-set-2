'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages



# Your code here

import pandas as pd
def preprocessing():
    """reads the cleaned data, performs merging, and creates necessary
    features for further analysis.

    Returns:
        df_arrests(dataframe): The merged and preprocessed dataframe.
    """


    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')

    #full outer join/merge on 'person_id'
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])


    def was_arrested_for_felony_within_year(row, df):
        """determine if there's a felony arrest in the next year
    
        Args:
        row(row of dataframe): row of the dataframe.
        df(dataframe): a dataframe 

        Returns:
            int: 1 if there is another felony in 1 year and 0 if not
        """
        arrest_date = row['arrest_date_univ']
        person_id = row['person_id']
        start_date = arrest_date + pd.Timedelta(days=1)
        end_date = arrest_date + pd.Timedelta(days=365)
        felony_arrests = df[(df['person_id'] == person_id) &
                        (df['arrest_date_event'] >= start_date) &
                        (df['arrest_date_event'] <= end_date) &
                        (df['charge_degree'] == 'felony')]
        return 1 if len(felony_arrests) > 0 else 0

    df_arrests['y'] = df_arrests.apply(was_arrested_for_felony_within_year, axis=1, df=df_arrests)

    share_rearrested = df_arrests['y'].mean()
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {share_rearrested:.2%}")


    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)

    share_current_felonies = df_arrests['current_charge_felony'].mean()
    print(f"What share of current charges are felonies? {share_current_felonies:.2%}")

    def num_felony_arrests_last_year(row, df):
        """creates predictive feature `num_fel_arrests_last_year`

        Args:
            row(row of dataframe): row of the dataframe.
            df(dataframe): a dataframe 

        Returns:
            len(felony_arrests): the number of felony arrests in the last year

        """
        arrest_date = row['arrest_date_univ']
        person_id = row['person_id']
        start_date = arrest_date - pd.Timedelta(days=365)
        end_date = arrest_date - pd.Timedelta(days=1)
        felony_arrests = df[(df['person_id'] == person_id) &
                        (df['arrest_date_event'] >= start_date) &
                        (df['arrest_date_event'] <= end_date) &
                        (df['charge_degree'] == 'felony')]
        return len(felony_arrests)

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(num_felony_arrests_last_year, axis=1, df=df_arrests)

    #calculates the average number of felony arrests in the last year
    average_felony_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_felony_arrests_last_year:.2f}")

    print(f"Mean of 'num_fel_arrests_last_year': {df_arrests['num_fel_arrests_last_year'].mean():.2f}")

    print(pred_universe.head())

    df_arrests.to_csv('data/df_arrests.csv', index=False)

    return df_arrests




