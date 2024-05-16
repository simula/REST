from pathlib import Path
import pandas as pd
import os
import gt3x


def gt3x_to_csv(path_to_folder: Path):
    for file_name in os.listdir(path_to_folder):
        if file_name.endswith('.gt3x'):
            actigraph_acc, actigraph_time, meta_data = gt3x.read_gt3x(path_to_folder / file_name)
            actigraph_acc = pd.DataFrame(actigraph_acc, columns=["x", "y", "z"])
            actigraph_acc.index = actigraph_time
            actigraph_acc.to_csv(path_to_folder / f"{file_name}.csv")
    print("Done")


def load_in_gt3x_files(path_to_folder: Path):
    files = {}
    for file_name in os.listdir(path_to_folder):
        if file_name.endswith('.gt3x'):
            print(f"START loading {file_name}")
            actigraph_acc, actigraph_time, meta_data = gt3x.read_gt3x(path_to_folder / file_name)
            print(f"END loading {file_name}")
            actigraph_acc = pd.DataFrame(actigraph_acc, columns=["x", "y", "z"])
            actigraph_acc.index = actigraph_time
            files[file_name] = actigraph_acc
    return files


def read_in_gt3x_csv(path_to_folder: Path):
    activities = {}
    for filename in os.listdir(path_to_folder)[:3]:
        if filename.endswith('.csv'):
            activities[filename[:-5]] = pd.read_csv(path_to_folder / filename, engine="python")
    return activities
