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
        "date_group": np.random.randint(5, size=300)})


def get_temporal_mock_graph_with_attributes():
    """
    Create temporal graph
    :return:
        pd.DataFrame
    """
    attr_dict = {}
    for i in range(3):
        attr_dict.update(
            {j+20*i: attributes for j, attributes in
             enumerate(np.random.randint(i, i + 5, size=[20, 5]))})

    return pd.DataFrame({
        "node_1": np.concatenate([
            np.random.randint(20, size=100),
            np.random.randint(20, 40, size=100),
            np.random.randint(40, 60, size=100)]),
        "node_2": np.concatenate([
            np.random.randint(20, size=100),
            np.random.randint(20, 40, size=100),
            np.random.randint(40, 60, size=100)]),
        "date_group": np.random.randint(5, size=300),

        "node_a": np.concatenate([
            np.random.randint(5, size=[5, 100]),
            np.random.randint(5, 10, size=[5, 100]),
            np.random.randint(10, 15, size=[5, 100])],  axis=1).T.tolist()
    }), attr_dict
