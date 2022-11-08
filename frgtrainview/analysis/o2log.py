import datetime
import pandas as pd

"""
Read from the o2 arduino running log and determine batch based on times
"""


LOG_FILE = 'PASCAL_ENV_2022-10-31.csv'

def slice_o2log(start_datetime, end_datetime):
    # extract the rows in the log between two times
    # structure is datetime, oxygen_ppm, humidity_ppm
    # date is in format yyyy/mm/dd hh:mm:ss (24-hour time, single digits not 0-padded)
    # roughly 3 seconds between each row

    # note: this may get less efficient as the log gets bigger, since whole file is loaded
    # try using the log interval to smartly search instead
    df = pd.read_csv(LOG_FILE, names=['datetime', 'oxygen_ppm', 'humidity_ppm'], parse_dates=[0])
    sliced = df[df['datetime'] >= start_datetime]
    sliced = sliced[sliced['datetime'] <= end_datetime]
    return sliced
