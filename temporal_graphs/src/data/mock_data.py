import pandas as pd
import numpy as np


def get_temporal_mock_graph():
    """
    Create temporal graph
    TODO: add options to create not random graph
    :return:
        pd.DataFrame
    """
    return pd.DataFrame({
        "node_1": np.random.randint(6, size=100),
        "node_2": np.random.randint(6, size=100),
        "date_group": np.random.randint(5, size=100)})
