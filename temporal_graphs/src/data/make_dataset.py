from typing import Optional, Union
import pandas as pd
from temporal_graphs.src.data import load_data
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_subreddit_links(prefix: Optional[str] = "../") -> pd.DataFrame:
    """

    :param prefix: : (str) applies to path (before) for using from different entry points
    :return: (pd.DataFrame)
    """
    df = load_data.load_snap_reddithyperlinks_body(prefix=prefix)
    df = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT']]
    return df


def add_date_group(df: pd.DataFrame,
                   date_column: Union[str, datetime.timestamp],
                   date_is_string: Optional[bool] = True,
                   window: Optional[int] = 6,
                   offset: Optional[int] = 3,
                   number_of_windows: Optional[int] = 10):
    """
    Add date groups to dataframe
    :param df: (pd.DataFrame)
    :param date_column: (str) - column where is a date (assuming is in this format 2017-04-30 16:41:53)
    :param date_is_string: (bool) whether the date is string
    :param window: (int) size of window
    :param offset: (int) size of offset (the next window will begin (previous window start + offset)
    :param number_of_windows: (int) number of windows
    :return:
    """
    if date_is_string:
        df[date_column] = df[date_column].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d').date())

    min_date_time = min(df[date_column])
    date_ranges = [[min_date_time + relativedelta(months=offset * i),
                    min_date_time + relativedelta(months=offset * i + window)]
                   for i in range(number_of_windows * window)]
    df["date_group"] = None
    for i, date_range in enumerate(date_ranges):
        df.loc[(df[date_column] >= date_range[0]) &
               (df[date_column] <= date_range[1]), "date_group"] = i
    return df
