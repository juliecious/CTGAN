import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from ctgan import load_demo
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from sklearn.model_selection import train_test_split
from utils import convert_adult_ds, eval_dataset, plot_scores

from sklearn.datasets import fetch_covtype
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    raw_data = fetch_covtype()
    data = np.concatenate((raw_data.data,
                           raw_data.target.reshape((raw_data.target.shape[0], 1))),
                          axis=1)
    columns = [
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Wilderness_Area1',
        'Wilderness_Area2',
        'Wilderness_Area3',
        'Wilderness_Area4',
        'Soil_Type1',
        'Soil_Type2',
        'Soil_Type3',
        'Soil_Type4',
        'Soil_Type5',
        'Soil_Type6',
        'Soil_Type7',
        'Soil_Type8',
        'Soil_Type9',
        'Soil_Type10',
        'Soil_Type11',
        'Soil_Type12',
        'Soil_Type13',
        'Soil_Type14',
        'Soil_Type15',
        'Soil_Type16',
        'Soil_Type17',
        'Soil_Type18',
        'Soil_Type19',
        'Soil_Type20',
        'Soil_Type21',
        'Soil_Type22',
        'Soil_Type23',
        'Soil_Type24',
        'Soil_Type25',
        'Soil_Type26',
        'Soil_Type27',
        'Soil_Type28',
        'Soil_Type29',
        'Soil_Type30',
        'Soil_Type31',
        'Soil_Type32',
        'Soil_Type33',
        'Soil_Type34',
        'Soil_Type35',
        'Soil_Type36',
        'Soil_Type37',
        'Soil_Type38',
        'Soil_Type39',
        'Soil_Type40',
        'Cover_Type']
    # data = pd.read_csv('/Users/juliefang/PycharmProjects/CTGAN/examples/csv/covertype.csv')
    target = 'Cover_Type'
    data = pd.DataFrame(data, columns=columns)
    # X = data.drop([target], axis=1)
    # y = data[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # data_train = np.concatenate((X_train,
    #                              y_train.values.reshape((y_train.values.shape[0], 1))),
    #                             axis=1)

    ctgan = DPCTGANSynthesizer(verbose=True, private=True, epochs=1, target_epsilon=1)
    ctgan.fit(data)
    ctgan.plot_losses(save=False)
