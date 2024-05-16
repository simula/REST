# REST

# Elite Female Football Athletes Dataset and Analysis Tools

This repository contains various scripts and tools for analyzing a novel dataset of 21 elite female football athletes. The dataset comprises 17 days of actigraphy, well-being, caffeine consumption, screen time, and daily hand strength test data. The aim is to provide a comprehensive understanding of the interplay between lifestyle factors, sleep, and athletic performance.


## Repository Structure
- `algorithms`: The folder contains several sleep detection algorithms and non wear algorithms. Additional, it provides sleep statistic functions and a base class to load in the actigraphy files. 
- `data_preprocessing`: Script for preprocessing and loading in of the original gt3x files. Addditional, a list of visualisations. 
- `generate_reports.py`: Example script how to read in actigraphy files and generate sleep statistics, sleep annotations and plots. 

- `technical_validation.py`: Script for performing technical validation of data or models. The script reproduces the figures and results of the paper. 

- `annonymisation.py`: Script for anonymizing sensitive data. It applies techniques to remove or obfuscate personally which we used for the annonymisation of the data. 

- `transfer_learning_inference.py`: Script for performing inference using transfer learning models. It applies pre-trained models to new datasets for prediction or classification tasks.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/simula/REST.git
    cd REST
    ```

2. Create and activate a virtual environment:

    ```bash
    conda env create -f environment.yml
    ```
## Usage

