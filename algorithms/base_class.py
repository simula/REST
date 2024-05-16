from pathlib import Path
from typing import Callable, Dict, List
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.sleep_statistics import get_sleep_stats
from algorithms.sleep_stage_algorithms import calculate_total_vector, merge_non_wear_sleep_stages
from data_preprocessing.visualisation import plot_activity_per_day_line, plot_activity_per_day,plot_activity_per_night_line


def apply_rescoring(sleep_stages, rescoring_rules: List[Callable]):
    if rescoring_rules:
        for rule in rescoring_rules:
            sleep_stages = rule(sleep_stages)
        return sleep_stages
    return sleep_stages


class SleepStageAlgo:
    def __init__(self, algorithm: Callable, non_wear_algorithm: Callable, statistics: pd.DataFrame,
    activity_stages: pd.Series, sleep_stages: pd.Series, non_wear_times: pd.Series):
        self.algorithm = algorithm
        self.non_wear_algorithm = non_wear_algorithm
        self.statistics = statistics
        self.activity_stages = activity_stages
        self.sleep_stages = sleep_stages
        self.non_wear_times = non_wear_times

    @classmethod
    def fit(cls, sleep_stage_algorithm: Callable, non_wear_algorithm: Callable, signals: pd.DataFrame,
            signal_name: Dict[str, str], rescoring_rules, epoch_duration, **kwargs):
        if signal_name["non_wear"] == "vm":
            signal_non_wear = calculate_total_vector(signals)
        elif signal_name["non_wear"] in ["x", "y", "z"]:
            signal_non_wear = signals[signal_name["non_wear"]]
        else:
            raise ValueError("Signal name doesn't exist")

        if signal_name["sleep_algo"] == "vm":
            signal_sleep = calculate_total_vector(signals)
        elif signal_name["sleep_algo"] in ["x", "y", "z"]:
            signal_sleep = signals[signal_name["sleep_algo"]]
            #signal_sleep["VM"] = calculate_total_vector(signals[["x", "y", "z"]])
        else:
            raise ValueError("Signal name doesn't exist")
        non_wear_time = non_wear_algorithm(signal_non_wear, **kwargs)
        sleep_stages = apply_rescoring(sleep_stage_algorithm(signal_sleep), rescoring_rules)
        #sleep_stages = sleep_stage_algorithm(signal_sleep, epoch_duration)
        merged_series = merge_non_wear_sleep_stages(non_wear_time, sleep_stages)
        statistics = get_sleep_stats(merged_series, signal_sleep)
        return cls(sleep_stage_algorithm, non_wear_algorithm, statistics, merged_series, sleep_stages, non_wear_time)

    def plot_activity_stages(self, path_to_figures: Path, activity: pd.Series, type: str, begin = None, end = None):
        if type == "dots":
            plot_activity_per_day(activity, self.activity_stages, path_to_figures)
        if type == "lines":
            plot_activity_per_day_line(activity, self.activity_stages, path_to_figures)
        if type == "night":
            plot_activity_per_night_line(activity, self.activity_stages, path_to_figures, begin, end)
        else:
            raise ValueError(f"Type: {type} is not available. Choose from ['dots', 'lines']")





