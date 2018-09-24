# encoding: utf-8
import logging
import pandas as pd
import numpy as np
from datetime import timedelta

logger = logging.getLogger(__name__)


def load_data_from_pkl(pkl_file):
    # need to format the timestamp column
    df = pd.read_pickle(pkl_file)
    y_label = [x for x in df.columns if x.lower().startswith('y')]
    if not bool(y_label):
        logger.warning('Y data missed.')
        x_df = df
        y_df = None
    else:
        y_df = df[y_label]
        x_df = df.drop(y_label, axis=1)
    return x_df, y_df
