from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import timedelta

from algorithms.base_class import SleepStageAlgo
from algorithms.non_wear_alogrithms import threshold_non_wear_times
from algorithms.sleep_stage_algorithms import calculate_oakley
from algorithms.sleep_stage_algorithms import rule_a, rule_b, rule_c, rule_d, rule_e
from algorithms.sleep_statistics import get_sleep_stats

def read_in_preprocess(path_to_file: Path):
    file = pd.read_csv(path_to_file, engine="python")
    # Assuming the first Friday date
    start_datetime = pd.to_datetime('2024-05-17 14:00:00')

    # Calculate the time delta
    time_delta = timedelta(seconds=30)

    # Generate the datetime series
    datetime_series = [start_datetime + i * time_delta for i in range(len(file))]

    # Assign the datetime series to a new column in the dataframe
    file.index = datetime_series
    file = file.drop(["Unnamed: 0"], axis=1)
    return file


def fit_model(file: pd.DataFrame, axis: str, baseline_period: int, epoch_length, min_length, rescoring_rules):
    if np.max(file[axis][:baseline_period]) > 10:
        baseline_non_wear_threshold = 6
    else:
        baseline_non_wear_threshold =np.max(file[axis][:baseline_period])
    model = SleepStageAlgo.fit(calculate_oakley, threshold_non_wear_times, file, {"sleep_algo": axis, "non_wear": axis},
                rescoring_rules , epoch_length, **{"min_length": min_length, "threshold": baseline_non_wear_threshold})
    return model


def generate_report(name, model: SleepStageAlgo, path_to_reports: Path, path_to_figures: Path):
    #begin_longest_period = model.statistics["start_date"]
    #end_longest_period = model.statistics["end_date"]
    #model.plot_activity_stages(path_to_figures / f"{name}_night_plot.png", activity, "night", begin_longest_period,
    #                          end_longest_period)
    model.statistics.to_csv(path_to_reports / f"{name}_report.csv")


def main():
    path_to_data = Path(__file__).parent / "data" / "actigraphy"
    path_to_reports = Path(__file__).parent / "data"
    path_to_figures = Path(__file__).parent / "figures" / "night"
    for i, file_name in enumerate(os.listdir(path_to_data)):    
        axis = "z"
        name = file_name.split(".")[0]
        print(f"Processing {name}")
        file = read_in_preprocess(path_to_data / file_name)
        model = fit_model(file, axis=axis, baseline_period=90, epoch_length=60, min_length=45,
                          rescoring_rules=[rule_a, rule_b, rule_c, rule_d, rule_e])
        generate_report(name, model, path_to_reports, path_to_figures)


if __name__ == "__main__":
    main()
