import pandas as pd
import numpy as np


def get_temporal_mock_graph():
    """
    Create temporal graph
    :return:
        pd.DataFrame
    """
    return pd.DataFrame({
        "node_1": np.concatenate([
            np.random.randint(20, size=100),
            np.random.randint(20, 40, size=100),
            np.random.randint(40, 60, size=100)]),
        "node_2": np.concatenate([
            np.random.randint(20, size=100),
            np.random.randint(20, 40, size=100),
            np.random.randint(40, 60, size=100)]),
        "date_group": np.random.randint(1, size=300)})
