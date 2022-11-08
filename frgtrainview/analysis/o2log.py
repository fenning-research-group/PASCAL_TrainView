from datetime import datetime, timedelta
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

def get_batch_o2log(outputdir, before_time=timedelta(0), after_time=timedelta(0)):
    """Returns the slice of the o2 log that contains the data that was recorded during a batch.

    Args:
        outputdir (path): a path to a batch's outputdir, which contains the .log file with timestamps.
        before_time (datetime.timedelta): a timedelta representing the amount of time to read logs prior to the start of the batch.
        after_time (datetime.timedelta): a timedelta representing the amount of time to read logs after the end of the batch.
    
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

    batch_start_time = ' '.join(first_line.split(' ')[:3])
    batch_end_time = ' '.join(last_line.split(' ')[:3])

    batch_start_time = datetime.strptime(batch_start_time, format)
    batch_end_time = datetime.strptime(batch_end_time, format)

    # now add the before_time and after_time
    start_time = batch_start_time - before_time
    end_time = batch_end_time + after_time

    # get the log data
    df = slice_o2log(start_time, end_time)

    # now add column for time since start of batch
    df['t'] = df['datetime'] - batch_start_time # normalize time relative to batch start
    df['t'] = df['t'].apply(lambda t: t.total_seconds()) # convert to seconds
    return df
