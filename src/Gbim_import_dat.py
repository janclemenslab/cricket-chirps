from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data/behavior"
COLUMNS = ("CDUR", "CPAU", "CPER", "CDC", "N", "rXY")


def _read(name):
    table = pd.read_csv(DATA / name, float_precision="round_trip")
    return [table[column].to_numpy() for column in COLUMNS]


def get_Gbim_data_chirpfield():
    return _read("chirpfield.csv")


def get_Gbim_data_DC():
    return _read("duty_cycle.csv")
