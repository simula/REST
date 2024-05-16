from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_activity_per_night_line(activity_series: pd.Series, state_series: pd.Series, path_to_figure: Path, start_time_period,
                                 end_time_period):
    # Define the colors for each state
    colors = {"n": "red", "w": "green", "s": "blue"}

    # Define start and end times
    start_time = pd.Timestamp("21:00:00").time()
    end_time = pd.Timestamp("11:00:00").time()

    # Filter for night time activity: 9pm to 11:30am
    activity_series = activity_series.between_time('21:00', '11:00')
    state_series = state_series.between_time('21:00', '11:00')

    # Create a new figure
    fig, axs = plt.subplots(len(activity_series.groupby(activity_series.index.date)),
                            figsize=(15, 5 * len(activity_series.groupby(activity_series.index.date))), sharey=True)

    # Adjust the logic to group by both dates (9pm of one day to 11:30am of the next day)
    def custom_grouper(timestamp):
        if timestamp.time() >= start_time:
            return timestamp.date()
        return (timestamp - pd.Timedelta('1 day')).date()

    activity_by_day = activity_series.groupby(activity_series.index.to_series().apply(custom_grouper))
    state_by_day = state_series.groupby(state_series.index.to_series().apply(custom_grouper))

    # Plot the activity data, colored by state for each day
    for i, ((day1, activity_day), (day2, state_day), start, end) in enumerate(zip(activity_by_day, state_by_day,
                                                                      start_time_period, end_time_period)):
        assert day1 == day2  # sanity check that the days match
        for state, color in colors.items():
            # Get the activity data for the current state
            activity_state = activity_day[state_day == state]
            # Plot vertical lines at each data point
            axs[i].vlines(x=activity_state.index, ymin=0, ymax=activity_state, color=color, label=state)
            axs[i].legend()
            if not pd.isnull(start) or not pd.isnull(end):
                axs[i].axvspan(start, end, alpha=0.05, color='blue')
            axs[i].set_title(f"Day: {day1}")
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Activity')
            axs[i].set_ylim(0, 1000)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            #axs[i].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))

            # Rotate x-axis labels to prevent overlap
            for label in axs[i].get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(path_to_figure)


def plot_activity_colored_by_state(activity_series, state_series, path_to_figure):
    # Define the colors for each state
    colors = {"n": "red", "w": "green", "s": "blue"}
    markers = {"n": "v", "w": ".", "s": "d"}

    # Create a new figure
    plt.figure(figsize=(15, 5))

    # Plot the activity data, colored by state
    for state, color in colors.items():
        # Get the activity data for the current state
        marker = markers[state]
        activity_state = activity_series[state_series == state]
        # Plot the activity data
        plt.scatter(activity_state.index, activity_state, color=color, label=state, marker=marker)

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel('Time')
    plt.ylabel('Activity')

    # Show the plot
    plt.savefig(path_to_figure)


def plot_activity_per_day_line(activity_series, state_series, path_to_figure):
    # Define the colors for each state
    colors = {"n": "red", "w": "green", "s": "blue"}

    # Separate data into days from 10am to 10am
    activity_by_day = activity_series.groupby(activity_series.index.date)
    state_by_day = state_series.groupby(state_series.index.date)

    # Create a new figure
    fig, axs = plt.subplots(len(activity_by_day), figsize=(15, 5 * len(activity_by_day)), sharey=True)

    # Plot the activity data, colored by state for each day
    for i, ((day1, activity_day), (day2, state_day)) in enumerate(zip(activity_by_day, state_by_day)):
        assert day1 == day2  # sanity check that the days match
        for state, color in colors.items():
            # Get the activity data for the current state
            activity_state = activity_day[state_day == state]
            # Plot vertical lines at each data point
            axs[i].vlines(x=activity_state.index, ymin=0, ymax=activity_state, color=color, label=state)
            axs[i].legend()
            axs[i].set_title(f"Day: {day1}")
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Activity')

    # Show the plot
    plt.tight_layout()
    plt.savefig(path_to_figure)


def plot_activity_per_day(activity_series, state_series, path_to_figure):
    # Define the colors for each state
    colors = {"n": "red", "w": "green", "s": "blue"}
    markers = {"n": "v", "w": ".", "s": "d"}

    # Separate data into days from 10am to 10am
    activity_by_day = activity_series.groupby(activity_series.index.date)
    state_by_day = state_series.groupby(state_series.index.date)

    # Create a new figure
    fig, axs = plt.subplots(len(activity_by_day), figsize=(15, 5 * len(activity_by_day)), sharey=True)

    # Plot the activity data, colored by state for each day
    for i, ((day1, activity_day), (day2, state_day)) in enumerate(zip(activity_by_day, state_by_day)):
        assert day1 == day2  # sanity check that the days match
        for state, color in colors.items():
            # Get the activity data for the current state
            marker = markers[state]
            activity_state = activity_day[state_day == state]
            # Plot the activity data on a separate subplot for each day
            axs[i].scatter(activity_state.index, activity_state, color=color, label=state, marker=marker)
            axs[i].legend()
            axs[i].set_title(f"Day: {day1}")
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Activity')

    # Show the plot
    plt.tight_layout()
    plt.savefig(path_to_figure)


def plot_with_confidence_interval(predicted_probs, index):
    # Calculate the upper and lower bounds for a 95% confidence interval
    # Assuming a Gaussian distribution, 1.96 is the Z-score for 95% CI
    lower_percentile = 5
    upper_percentile = 95

    means = predicted_probs.mean(axis=0)
    std_devs = predicted_probs.std(axis=0)

    lower_bound = np.clip(np.percentile(predicted_probs - std_devs, lower_percentile, axis=0), 0, 1)
    upper_bound = np.clip(np.percentile(predicted_probs + std_devs, upper_percentile, axis=0), 0, 1)

    time_stamps = np.arange(len(means))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(index, means, color='blue', label="Predicted Mean")
    plt.fill_between(index, lower_bound, upper_bound, color='gray', alpha=0.5, label="95% Confidence Interval")
    plt.title("Predicted Time Series with Confidence Intervals")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
