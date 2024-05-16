import pandas as pd
import numpy as np

from algorithms.non_wear_alogrithms import mean_abs_diff
from algorithms.sleep_stage_algorithms import rule_a, rule_b, rule_c, rule_d, rule_e

def apply_rescoring(sleep_stages, rescoring_rules):
    if rescoring_rules:
        for rule in rescoring_rules:
            sleep_stages = rule(sleep_stages)
        return sleep_stages
    return sleep_stages

def split_into_days(series):
    series_day = series.resample('24H', offset="14h", label='left')
    return series_day


def sleep_time_per_day(series_day):
    sleep_time = {}
    for day, activities in series_day:
        sleep_start = day + pd.DateOffset(hours=12)
        sleep_end = day + pd.DateOffset(hours=24)
        sleep_period = activities[
            (activities.index >= sleep_start) & (activities.index <= sleep_end) & (activities == 's')]

        # Count the total sleep time
        sleep_time[day] = len(sleep_period)/60

    # Return the sleep time as a pandas series
    return pd.Series(sleep_time)


def transform_series(series):
    def transform_char(char):
        if char == 's':
            return 1
        else:
            return 0
    transformed_series = series.map(transform_char)
    return transformed_series


def filter_for_night_period(day, activities):
    sleep_start = day + pd.DateOffset(hours=7)
    sleep_end = day + pd.DateOffset(hours=14+7)
    sleep_activities = activities[
        (activities.index >= sleep_start) & (activities.index <= sleep_end)]
    sleep_activities = transform_series(sleep_activities)
    return sleep_activities


def adjust_series_endpoints(s):
    if s.iloc[0] != 0:
        s = pd.concat([pd.Series([0], index=[s.index[0] - pd.Timedelta(minutes=1)]), s])
    if s.iloc[-1] != 0:
        s = pd.concat([s, pd.Series([0], index=[s.index[-1] + pd.Timedelta(minutes=1)])])
    return s

def apply_rolling_mean_rescore(series, window_size):
    """
    Rescore a series based on the rolling mean over a specified window size.
    Any epoch where the rolling mean is above 0.5 is set to 1, otherwise 0.

    Parameters:
    - series: Pandas Series to rescore.
    - window_size: The number of epochs over which to calculate the rolling mean.

    Returns:
    - Pandas Series with the same length as the input series, after applying the rescoring rule.
    """
    # Calculate rolling mean with min_periods=1 to ensure the result is the same length as the input
    rolling_mean = series.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Rescore based on the rolling mean
    rescored_series = (rolling_mean > 0.5).astype(int)
    
    return rescored_series

def extract_sleep_intervals(sleep):
    s = apply_rolling_mean_rescore(sleep, 15)
    start_flag = (s.diff() == 1)
    end_flag = (s.diff() == -1)
    start_dates = s.index[start_flag]
    end_dates = s.index[end_flag]

    # If the series starts with a sleep state, add a start at the beginning
    if len(end_dates) > 0 and (len(start_dates) == 0 or end_dates[0] < start_dates[0]):
        start_dates = pd.Index([s.index[0]]).append(start_dates)
    
    # If the series ends with a sleep state, add an end at the end
    if len(start_dates) > len(end_dates) or (len(start_dates) > 0 and len(end_dates) == 0):
        end_dates = end_dates.append(pd.Index([s.index[-1]]))


    # Proceed only if there are valid intervals to process
    if len(start_dates) > 0 and len(end_dates) > 0:
        df_intervals = pd.DataFrame({'start_date': start_dates, 'end_date': end_dates})
        df_intervals['length'] = df_intervals['end_date'] - df_intervals['start_date']
        return df_intervals
    else:
        # Return an empty DataFrame with the appropriate columns if no intervals
        return pd.DataFrame(columns=['start_date', 'end_date', 'length'])

def extract_sleep_intervals_(sleep):
    s = apply_rolling_mean_rescore(sleep, 10)
    start_flag = (s.diff() == 1)
    end_flag = (s.diff() == -1)
    start_dates = s.index[start_flag]
    end_dates = s.index[end_flag]
    # Ensure the first element is not a sleep state (assuming '0' is sleep, '1' is wake)
    if s.iloc[0] == 0:
        start_dates = start_dates.insert(0, s.index[0])

    # Ensure the last element returns to wakefulness to close the last sleep period
    if s.iloc[-1] == 0:
        end_dates = end_dates.append(pd.Index([s.index[-1]]))
    df_intervals = pd.DataFrame({'start_date': start_dates, 'end_date': end_dates})
    df_intervals['length'] = df_intervals['end_date'] - df_intervals['start_date']
    return df_intervals

def get_longest_interval(df_intervals):
    nr_sleeping_episodes = len(df_intervals['length'])
    df_intervals.loc[:, "total_sleep_time"] = df_intervals['length'].sum()
    df_intervals = df_intervals[df_intervals['length'] == df_intervals['length'].max()].copy().iloc[0:1]
    df_intervals.loc[:, "sleep_intervals"] = nr_sleeping_episodes
    if df_intervals.empty:
        df_intervals = pd.DataFrame({
            "start_date": [np.nan],
            "end_date": [np.nan],
            "length": [np.nan],
            "total_sleep_time": [np.nan],
            "sleep_intervals": [np.nan],
        })
    return df_intervals


def extract_sleep_statistics_(series_day, activity_series):
    longest_sleep_periods = []
    sleep_period_statistics = []
    days = []
    for day, activities in series_day:
        sleep_activities = filter_for_night_period(day, activities)
        adjusted_series = adjust_series_endpoints(sleep_activities)

        rescored_series = apply_rescoring(adjusted_series, [rule_a, rule_b, rule_c,rule_d, rule_e])
        sleep_intervals = extract_sleep_intervals(rescored_series)
        if sleep_intervals.empty:
            longest_sleep_periods.append(pd.DataFrame({
                "start_date": [np.nan],
                "end_date": [np.nan],
                "length": [np.nan],
                "total_sleep_time": [np.nan],
                "sleep_intervals": [np.nan],
            }))
            sleep_period_statistics.append({"sleep period mean": np.nan,
                                            "sleep period standard deviation": np.nan,
                                            "sleep period msd": np.nan
                                            })
        else:
            longest_interval = get_longest_interval(sleep_intervals)
            if longest_interval.empty:
                sleep_period_statistics.append({"sleep period mean": np.nan,
                "sleep period standard deviation": np.nan,
                "sleep period msd": np.nan
                })
            else:
                sleep_period_statistics.append(get_statistics_on_sleep_period(longest_interval["start_date"],
                                                                 longest_interval["end_date"],
                                                                 activity_series))
            longest_sleep_periods.append(longest_interval)
        days.append(day)
    sleep_period_statistics_df = pd.DataFrame(sleep_period_statistics)
    longest_sleep_df = pd.concat(longest_sleep_periods, axis=0).reset_index(drop=True)
    result_df = pd.concat([longest_sleep_df, sleep_period_statistics_df], axis=1)
    result_df.index = days
    return result_df


def get_statistics_on_sleep_period(begin_date, end_date, activity):
    sleep_period = activity.loc[begin_date: end_date].reset_index(drop=True)
    return {"sleep period mean": np.mean(sleep_period),
            "sleep period standard deviation": np.std(sleep_period),
            "sleep period msd": mean_abs_diff(sleep_period)
            }


def get_sleep_stats(series, activity_series) -> pd.DataFrame:
    series_day = split_into_days(series)
    sleep_statistics_df = extract_sleep_statistics(series_day, activity_series)
    return sleep_statistics_df

def extract_sleep_statistics__(series_day, activity_series):
    sleep_statistics = []
    days = []
    for day, activities in series_day:
        sleep_activities = filter_for_night_period(day, activities)
        adjusted_series = adjust_series_endpoints(sleep_activities)
        rescored_series = apply_rescoring(adjusted_series, [rule_a, rule_b, rule_c, rule_d, rule_e])
        sleep_intervals = extract_sleep_intervals(rescored_series)

        if not sleep_intervals.empty:
            # Use the first interval for onset and the last interval for offset
            first_interval = sleep_intervals.iloc[0]
            last_interval = sleep_intervals.iloc[-1]

            # Identify the longest sleep interval
            sleep_intervals['interval_duration'] = (sleep_intervals['end_date'] - sleep_intervals['start_date']).dt.total_seconds() / 60
            longest_interval = sleep_intervals.loc[sleep_intervals['interval_duration'].idxmax()]

            # Statistics for the longest interval
            longest_interval_start = longest_interval['start_date']
            longest_interval_end = longest_interval['end_date']
            longest_interval_duration = longest_interval['interval_duration']

            # Aggregate sleep data
            sleep_onset = first_interval['start_date']
            sleep_offset = last_interval['end_date']
            total_sleep_time = sleep_intervals['interval_duration'].sum()
            number_of_intervals = len(sleep_intervals)

            sleep_statistics.append({
                "sleep_onset": sleep_onset,
                "sleep_offset": sleep_offset,
                "total_sleep_time": total_sleep_time.round(3),
                "sleep_intervals": number_of_intervals,
                "longest_interval_start": longest_interval_start,
                "longest_interval_end": longest_interval_end,
                "longest_interval_duration": longest_interval_duration.round(3),
                # Additional statistics for the longest sleep interval could be added here
            })
        else:
            # Append NaN values or placeholders if no intervals were found
            sleep_statistics.append({
                "sleep_onset": np.nan,
                "sleep_offset": np.nan,
                "total_sleep_time": np.nan,
                "sleep_intervals": 0,
                "longest_interval_start": np.nan,
                "longest_interval_end": np.nan,
                "longest_interval_duration": np.nan,
                # Placeholders for additional longest interval statistics
            })
        
        days.append(day)
    
    sleep_statistics_df = pd.DataFrame(sleep_statistics, index=days)
    return sleep_statistics_df

def extract_sleep_statistics(series_day, activity_series):
    sleep_statistics = []
    days = []
    
    for day, activities in series_day:
        sleep_activities = filter_for_night_period(day, activities)
        adjusted_series = adjust_series_endpoints(sleep_activities)
        rescored_series = apply_rescoring(adjusted_series, [rule_a, rule_b, rule_c, rule_d, rule_e])
        sleep_intervals = extract_sleep_intervals(rescored_series)

        if not sleep_intervals.empty:
            # Calculate first and last interval for onset and offset
            first_interval = sleep_intervals.iloc[0]
            last_interval = sleep_intervals.iloc[-1]
            sleep_intervals['interval_duration'] = (sleep_intervals['end_date'] - sleep_intervals['start_date']).dt.total_seconds() / 60
            longest_interval = sleep_intervals.loc[sleep_intervals['interval_duration'].idxmax()]
            
            # Aggregate sleep data
            aggregated_data = {
                "sleep_onset": first_interval['start_date'],
                "sleep_offset": last_interval['end_date'],
                "total_sleep_time": sleep_intervals['interval_duration'].sum().round(3),
                "sleep_intervals": len(sleep_intervals),
                "longest_interval_start": longest_interval['start_date'],
                "longest_interval_end": longest_interval['end_date'],
                "longest_interval_duration": longest_interval['interval_duration'].round(3),
            }

            # Calculate additional statistics for the longest sleep period
            sleep_period_stats = get_statistics_on_sleep_period(longest_interval['start_date'], longest_interval['end_date'], activity_series)
            aggregated_data.update(sleep_period_stats)
            
        else:
            # Placeholder values if no sleep intervals were found
            aggregated_data = {
                "sleep_onset": np.nan,
                "sleep_offset": np.nan,
                "total_sleep_time": np.nan,
                "sleep_intervals": 0,
                "longest_interval_start": np.nan,
                "longest_interval_end": np.nan,
                "longest_interval_duration": np.nan,
                "sleep period mean": np.nan,
                "sleep period standard deviation": np.nan,
                "sleep period msd": np.nan
            }

        sleep_statistics.append(aggregated_data)
        days.append(day)
    
    sleep_statistics_df = pd.DataFrame(sleep_statistics, index=days)
    return sleep_statistics_df
