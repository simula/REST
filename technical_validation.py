import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
import os

def create_activity_heatmap(data, path_to_figures, time_col='Date', activity_col='vm'):
    """
    Creates a heatmap of activity levels organized by time within a day and sequential days.

    Parameters:
    data (pandas.DataFrame): DataFrame containing the activity data.
    time_col (str): The name of the column in 'data' that contains the datetime information.
    activity_col (str): The name of the column in 'data' that contains the activity levels.
    """
    # Ensure the time column is in datetime format
    print(data.columns)
    data[time_col] = pd.to_datetime(data[time_col])
    
    # Extract time and date components
    data['time'] = data[time_col].dt.time
    data['date'] = data[time_col].dt.date
    
    # Pivot the data to get a 2D matrix of activity levels
    activity_matrix = data.pivot_table(index='date', columns='time', values=activity_col, aggfunc=np.mean)
    
    # Plot the heatmap
    plt.figure(figsize=(15, 7))
    sns.heatmap(activity_matrix, cmap='inferno', cbar_kws={'label': 'Activity Level'})
    plt.title('Activity Heatmap by Time of Day and Sequential Days')
    plt.xlabel('Time of Day')
    plt.ylabel('Sequential Days')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path_to_figures /"activity_heatmap.pdf")

def plot_24_hour_activity_profile(df, path_to_figures, datetime_col='time', activity_col='vm'):
    # Ensure the datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract the hour of day from the datetime
    df['hour_of_day'] = df[datetime_col].dt.hour

    # Group by hour of day and calculate the mean and standard deviation of activity level for each hour
    hourly_activity_mean = df.groupby('hour_of_day')[activity_col].mean()
    hourly_activity_std = df.groupby('hour_of_day')[activity_col].std()

    # Calculate upper and lower bounds for the std deviation band, ensuring it doesn't go below 0
    upper_bound = hourly_activity_mean + hourly_activity_std
    lower_bound = np.maximum(hourly_activity_mean - hourly_activity_std, 0)

    # Plot the 24-hour activity profile with std deviation band
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_activity_mean.index, hourly_activity_mean.values, color='black', label='Mean Activity Level')
    
    # Add standard deviation band in grey
    plt.fill_between(hourly_activity_mean.index, lower_bound, upper_bound, color='grey', alpha=0.5)
    
    
    # Determine the plot limits
    ymin, ymax = plt.ylim()
    # Add standard deviation shading in red below the mean
    plt.fill_between(hourly_activity_mean.index, lower_bound, hourly_activity_mean, color='red', alpha=0.5)
    
    # Fill the entire area under the mean down to 0 also in red but with less opacity
    plt.fill_between(hourly_activity_mean.index, 0, hourly_activity_mean, color='red', alpha=0.3)
    

    plt.fill_between(hourly_activity_mean.index, hourly_activity_mean, ymax, where=(hourly_activity_mean<=ymax), interpolate=True, color='blue', alpha=0.3)

    plt.fill_between(hourly_activity_mean.index, np.maximum(hourly_activity_mean, ymin), hourly_activity_mean, where=(hourly_activity_mean>=ymin), interpolate=True, color='red', alpha=0.3)
    
    #plt.title('24-Hour Activity Profile with Standard Deviation')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Activity Level')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_to_figures / "24_activity_profile.pdf")


def plot_spearman_heatmap(df, path_to_figures, columns):
    """
    Plots a heatmap of the Spearman rank correlation matrix for selected columns in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The dataset containing the variables.
    columns (list): List of column names to include in the correlation analysis.
    """
    # Select specified columns for correlation analysis
    df_selected = df[columns]
    
    # Calculate the Spearman rank correlation matrix
    spearman_corr = df_selected.corr(method='spearman')
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(16, 14))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(spearman_corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5, "label": 'Correlation'}, annot=True, fmt=".1f",
                annot_kws={"size": 14})  # Increase annotation font size
    
    # Add title and format
    plt.xticks(fontsize=16, rotation=45, ha='right')  # Increase x-axis labels font size and rotate for readability
    plt.yticks(fontsize=16, rotation=0)  # Increase y-axis labels font size
    plt.tight_layout()
    
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=14)  # Increase color bar tick labels font size
    
    # Show plot
    plt.savefig(path_to_figures/ "spearman-correlation.pdf")

def plot_trajectories(data, metric1, metric2):
    """
    Plots trajectories for two chosen metrics for each subject in the dataset.

    :param data: pandas DataFrame containing the longitudinal data
    :param metric1: String, the first metric to plot (column name from the DataFrame)
    :param metric2: String, the second metric to plot (column name from the DataFrame)
    """
    # Ensure 'date' is in datetime format for plotting
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
    
    # Set up the figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # Unique subjects
    subjects = data['id'].unique()
    
    # Plot each metric for each subject
    for subject in subjects:
        subject_data = data[data['id'] == subject]
        axes[0].plot(subject_data['date'], subject_data[metric1], label=f'Subject {subject}')
        axes[1].plot(subject_data['date'], subject_data[metric2], label=f'Subject {subject}')
    
    # Set the titles and legends
    axes[0].set_title(f'Trajectories of {metric1}')
    axes[1].set_title(f'Trajectories of {metric2}')
    #xes[0].legend()
    #axes[1].legend()
    axes[1].set_xlabel('Date')
    # Rotate date labels for clarity
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def ordinal_relation(df, x_col, y_col):
    subset = df[[x_col, y_col]]
    subset = subset.dropna()  # Remove missing values
    
    # Ensure data is numeric
    x = subset[x_col]
    y = subset[y_col]

    # Calculate the Spearman rank correlation
    rho, p_value = stats.spearmanr(x, y)

    # Calculate the best fit line for visual aid (even though Spearman's is non-linear)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    line = slope * x + intercept

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker=".", label='Original data')  # Changed to plt.scatter for clarity
    plt.plot(x, line, 'r', label='Fitted line')  # Best fit line remains for reference
    plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Correlation between {x_col} and {y_col} with Spearman rho = {rho:.2f}')
    plt.show()

def mixed_effects_model_with_time(df, independent_vars, dependent_var, participant_col, date_col):
    """
    Perform mixed effects model analysis accounting for both fixed and random effects,
    and including time (date) as a random effect for each participant.
    
    Parameters:
        df (pandas.DataFrame): The dataset containing the variables.
        independent_vars (list): List of columns in the dataframe representing independent (fixed) variables.
        dependent_var (str): The column representing the dependent (outcome) variable.
        participant_col (str): The column representing participant IDs for random effects.
        date_col (str): The column representing the time variable (e.g., days).
    """
    cols_to_check = independent_vars + [dependent_var, participant_col]
    df_cleaned = df.dropna(subset=cols_to_check)
    fixed_effects_formula = ' + '.join([f'C({var})' if df[var].dtype == 'object' else var for var in independent_vars])
    formula = f'{dependent_var} ~ {fixed_effects_formula}'#+ (0 + {date_col} | {participant_col})'
    
    # Fit the model
    model = smf.mixedlm(formula, data=df_cleaned, re_formula="1", groups=df_cleaned[participant_col]).fit()
    
    # Print the results
    print("Mixed Effects Model Results with Time Effect")
    print("-------------------------------------------")
    print(model.summary())

def correct_date_shift(dataframe, columns_to_shift):
    # Ensure 'date' is a datetime object if it's not already (optional based on your DataFrame)
    dataframe['date'] = pd.to_datetime(dataframe['date'], format='%d/%m/%Y')
    
    dataframe.sort_values(by=['id', 'date'], inplace=True)
    # Create a copy to avoid SettingWithCopyWarning and ensure changes are kept
    df_copy = dataframe.copy()
    
    for column in columns_to_shift:
        # Apply a groupby operation on 'id' and then shift the columns
        df_copy[column] = df_copy.groupby('id')[column].shift(-1)  # Use -1 to shift back by one day
    
    # Remove the last row for each 'id' after shifting
    df_copy = df_copy.groupby('id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    
    return df_copy.reset_index(drop=True)

def combine_and_create_heatmap(player_dfs, path_to_figures, datetime_col='time', activity_col='vm'):
    """
    Combines player DataFrames and creates a heatmap of average daily activity patterns for each player.
    """
    # Combine all player DataFrames into one DataFrame
    combined_df = pd.concat(player_dfs.values(), ignore_index=True)
    
    # Ensure the datetime column is in datetime format
    combined_df[datetime_col] = pd.to_datetime(combined_df[datetime_col])
    
    # Extract the hour of day from the datetime
    combined_df['hour_of_day'] = combined_df[datetime_col].dt.hour
    
    # Pivot the data for heatmap format
    heatmap_data = combined_df.pivot_table(values=activity_col, 
                                           index='player_id', 
                                           columns='hour_of_day', 
                                           aggfunc='mean')
    
    # Create figure and plot heatmap
    plt.figure(figsize=(15, len(heatmap_data.index) * 0.5))  # Adjust height based on the number of players
    ax = sns.heatmap(heatmap_data, cmap='inferno', cbar_kws={'label': 'Average Activity Level'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # Adjust tick size for color bar
    cbar.set_label('Average Activity Level', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Hour of Day', fontsize=20)
    plt.ylabel('Player', fontsize=20)
    plt.tight_layout()
    plt.savefig(path_to_figures / "average_daily_patterns.pdf")


def main():
    path_to_daily_reports = Path(__file__).parent / "data" / "daily_responses.csv"
    path_to_figures = Path(__file__).parent / "figures"
    path_to_example_activity = Path(__file__).parent / "data" / "actigraphy"
    player_dfs = {}
    for file in os.listdir(path_to_example_activity):
        if "secs" in file:
            continue
        name = file.split(".")[0][-3:]
        player_dfs[name] = pd.read_csv(path_to_example_activity/ file, delimiter=",")
        player_dfs[name]['player_id'] = f'Player_{name}'
    combine_and_create_heatmap(player_dfs, path_to_figures,'time', 'vm')
    example_1 = pd.read_csv(path_to_example_activity / "a2d89dcf-6bd1-445b-bc9c-7ac6b852a9fc.csv", delimiter=",")
    data = pd.read_csv(path_to_daily_reports, delimiter=";")
    data["matchday_bin"] = data["matchday"].isna()*1
    data = data.rename(columns={"soreness_locasion_1": "soreness_location_1"})
    shift_cols = ["fatigue", "soreness", "readiness", "caffeine_time", "sleepquality",  "sleepdura_sr"]
    #data = correct_date_shift(data, shift_cols)
    data['sleepdura_sr'] = data['sleepdura_sr'] * 60
    data['sleepquality'] = data['sleepquality'] - data['sleepquality'].mean()
    data['sleepquality_quadratic'] = data['sleepquality']**2
    data["handgrip_kg"] = data["handgrip_kg"].apply(lambda x: x if isinstance(x,float) else float(x.replace(",",".")))
    plot_24_hour_activity_profile(example_1, path_to_figures)
    plot_spearman_heatmap(data, path_to_figures, ['total_sleep_time',
       'sleep_intervals',
       'longest_interval_duration', 'sleep period mean',
       'sleep period standard deviation', 'sleep period msd', 'sleepquality',
        'matchday_bin', 'handgrip_kg', 'fatigue', 'soreness',
       'soreness_location_1', 'soreness_location_2', 'readiness',
       'sleepdura_sr', 'caffeine_mg'])
    
    mixed_effects_model_with_time(data, ["sleepdura_sr", "sleepquality", "sleepquality_quadratic"], "total_sleep_time", "id", "date_ordinal")

if __name__ == "__main__":
    main()