import datetime
import pandas as pd
import glob
import os

"""
Read from the o2 arduino running log and determine batch based on times
"""


LOG_FILE = 'PASCAL_ENV_2022-10-31.csv'

def slice_o2log(start_datetime, end_datetime, o2log_file=LOG_FILE):
    """Slice the o2 log and returns a pandas DataFrame.

    Args:
        start_datetime (datetime): a datetime object to begin the slice.
        end_datetime (datetime): a datetime object to end the slice.
        o2log_file (path): a path to the o2 log file to use, if not using the default.
    
    Returns:
        pd.DataFrame: contains the rows of the o2 log that are within the start and end datetimes.
    """
    # extract the rows in the log between two times
    # structure is datetime, oxygen_ppm, humidity_ppm
    # date is in format yyyy/mm/dd hh:mm:ss (24-hour time, single digits not 0-padded)
    # roughly 3 seconds between each row

    # note: this may get less efficient as the log gets bigger, since whole file is loaded
    # try using the log interval to smartly search instead
    df = pd.read_csv(o2log_file, names=['datetime', 'oxygen_ppm', 'humidity_ppm'], parse_dates=[0])
    sliced = df[df['datetime'] >= start_datetime]
    sliced = sliced[sliced['datetime'] <= end_datetime]
    return sliced

def get_batch_o2log(outputdir):
    """Returns the slice of the o2 log that contains the data that was recorded during a batch.

    Args:
        outputdir (path): a path to a batch's outputdir, which contains the .log file with timestamps.
    
    Returns:
        pd.DataFrame: contains the rows of the o2 log that were recorded during the batch.
        
    """

    # find the .log file for timestamps of when batch started and ended
    # start time is on first row, end time is on last row
    output_logfile = glob.glob(os.path.join(outputdir, '*.log'))[0]
    
    first_line, last_line = None, None
    with open(output_logfile) as file:
        # not the most efficient since all lines are read in, but works for small logs
        lines = file.readlines()
        first_line, last_line = lines[0], lines[-1]

    # now extract the times
    # in format mm/dd/yyyy hh:mm:ss AM/PM
    # day/time is 0-padded
    format = "%m/%d/%Y %I:%M:%S %p"

    start_time = ' '.join(first_line.split(' ')[:3])
    end_time = ' '.join(last_line.split(' ')[:3])

    start_time = datetime.datetime.strptime(start_time, format)
    end_time = datetime.datetime.strptime(end_time, format)

    return slice_o2log(start_time, end_time)
