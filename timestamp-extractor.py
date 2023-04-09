import os
import re
import statistics
import sys
from datetime import datetime
from re import Pattern
from typing import AnyStr, Union

# Путь к log-файлу передается в параеметре командной строки.
# Путь может содержать пробелы в имени папок и самого файла.

NORMAL_STR = "+"
CRITICAL_STR = "!"
WARNING_STR = "-"
EVENT_BODY_OFFSET = 32


def extract_token(source: str, regex_pattern: Pattern[AnyStr]) -> Union[str, None]:
    token_iter = re.finditer(regex_pattern, source)
    try:
        token_match_obj = token_iter.__next__()
        if token_match_obj is not None:
            result_str = token_match_obj.group(0)
            return result_str
    except StopIteration:
        return None


def classify_duration(event_duration: float, min_bound: float, max_bound: float, median: float, average: float) -> str:
    if event_duration <= 2 * median:
        return NORMAL_STR
    else:
        if (event_duration > 2 * median) and (event_duration <= 1.5*average):
            return WARNING_STR
        else:
            return CRITICAL_STR


if __name__ == '__main__':
    fileName = " ".join(sys.argv[1:])
    if not os.path.isfile(fileName):
        print(f"Can't open file: {fileName}")
        exit(1)

    f = open(file=fileName, mode="r")
    logContent = f.readlines()

    timestamp_pattern = re.compile("^\\[.[0-9\\- :,]+]")
    event_origin_pattern = re.compile("{[\\w.-]+}")
    event_data_pattern = re.compile(" - .+$")

    timestamp_min = None
    timestamp_duration = 0

    timestamp_vector = []
    durations_vector = []
    event_origin_vector = []
    event_data_vector = []

    for log_line in logContent:
        if len(log_line) == 0:
            continue
        timestamp_str = extract_token(log_line, timestamp_pattern)
        if timestamp_str is not None:
            timestamp_vector.append(timestamp_str)
            datetime_obj = datetime.strptime(timestamp_str[1:-1], '%Y-%m-%d %H:%M:%S,%f')
            if timestamp_min is None:
                timestamp_min = datetime_obj
                durations_vector.append(0)  # first event has no duration
            else:
                timestamp_diff = datetime_obj - timestamp_min
                timestamp_duration = timestamp_diff.total_seconds()
                durations_vector.append(timestamp_duration)
                timestamp_min = datetime_obj
        else:
            continue

        event_origin_str = extract_token(log_line[EVENT_BODY_OFFSET:-1], event_origin_pattern)
        event_origin_vector.append(event_origin_str if event_origin_str is not None else "No source")

        event_data_str = extract_token(log_line[EVENT_BODY_OFFSET:-1], event_data_pattern)
        if event_data_str is not None:
            if len(event_data_str) > 200:
                event_data_str = event_data_str[:200]
            event_data_vector.append(event_data_str[3:])
        else:
            event_data_vector.append("Error in log parsing format/criteria")

    normalized_duration_vector = durations_vector[1:]  # first event has default zero-duration
    duration_min = min(normalized_duration_vector)
    duration_max = max(normalized_duration_vector)
    duration_avg = statistics.mean(normalized_duration_vector)
    duration_med = statistics.median(normalized_duration_vector)

    print("\n   #\tTimestamp\t\t\t\t\t  Duration(sec)\tEvent origin (object)\t\t\t\t\t\t\t\tEvent data")
    print("----------------------------------------------------------------------------------------------------------"
          "-----------------------------------------------------------------")

    log_items_count = len(durations_vector)
    for line_num in range(log_items_count):
        timestamp = timestamp_vector[line_num]
        perf_marker = classify_duration(durations_vector[line_num], duration_min, duration_max, duration_med, duration_avg)
        duration = durations_vector[line_num]
        event_origin = event_origin_vector[line_num]
        event_data = event_data_vector[line_num]

        print(f"{line_num + 1: 4}\t{timestamp}\t{duration: 7.3f}\t{perf_marker}\t\t{event_origin}\t{event_data}")

    print("----------------------------------------------------------------------------------------------------------"
          "-----------------------------------------------------------------")

    print("\nSTATISTICS:\n")
    print(f"Number of events - {len(durations_vector)} items")
    print(f"Total duration\t - {sum(durations_vector):7.3f}")
    print(f"\t\t\tmin. - {duration_min:7.3f}")
    print(f"\t\t\tmax. - {duration_max:7.3f}")
    print(f"\t\t\tavg. - {duration_avg:7.3f}")
    print(f"\t\t\tmed. - {duration_med:7.3f}")
