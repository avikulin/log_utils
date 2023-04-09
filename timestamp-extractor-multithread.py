from __future__ import annotations
import os
import re
import statistics
import sys
import uuid
from datetime import datetime
from enum import Enum
from re import Pattern
from typing import AnyStr, Union, List, Dict, Iterable, Set, TextIO

# Путь к log-файлу передается в параеметре командной строки.
# Путь может содержать пробелы в имени папок и самого файла.

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TIMESTAMP_PATTERN = re.compile("^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}.\\d{3})")
EVENT_TYPE_PATTERN = re.compile(".{23}(INFO|ERROR|TRACE|DEBUG) \\d")
EVENT_THREAD_PATTERN = re.compile("\\d{4,} -{3} \\[([^:]+)]")
EVENT_ORIGIN_PATTERN = re.compile("\\d{4,} -{3} \\[[^:]+] (.+)\\s+: ")
EVENT_DATA_PATTERN = re.compile("\\d{4,} -{3} \\[[^:]+] .+\\s+: (.+)$")


class PerformanceMarker(Enum):
    NORMAL = "+"
    CRITICAL = "!"
    WARNING = "-"
    UNKNOWN = "?"

    def __repr__(self):
        return self.name


class TokenType(Enum):
    TIMESTAMP = "TIMESTAMP"
    STRING = "STRING"
    EVENT_TYPE = "EVENT_TYPE"


class EventType(Enum):
    INFO = "INFO"
    ERROR = "ERROR"
    TRACE = "TRACE"
    DEBUG = "DEBUG"

    def __repr__(self):
        return self.name

    @staticmethod
    def from_string(source: str):
        for event_type in EventType:
            if source == event_type.name:
                return event_type
        raise ValueError(f"'{source}' has no matches with elements of 'EventType' enum")


class Event:
    def __init__(self, timestamp: datetime, event_type: EventType, thread_name: str, origin: str, data: str):
        if timestamp is None or event_type is None or not thread_name or not origin or not data:
            raise ValueError("All method parameters are mandatory and must not null or empty")

        self.__id: uuid = uuid.uuid4()
        self.__perf_marker: PerformanceMarker = PerformanceMarker.UNKNOWN
        self.__timestamp: datetime = timestamp
        self.__duration: float = 0
        self.__event_type: EventType = event_type
        self.__origin: str = origin
        self.__thread: str = thread_name
        self.__data: str = data

    @property
    def id(self) -> uuid:
        return self.__id

    @property
    def timestamp(self) -> datetime:
        return self.__timestamp

    @property
    def duration(self) -> float:
        return self.__duration

    @duration.setter
    def duration(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Duration must be positive value. Passed '{value}' is incorrect")
        self.__duration = value

    @property
    def perf_marker(self) -> PerformanceMarker:
        return self.__perf_marker

    @perf_marker.setter
    def perf_marker(self, value: PerformanceMarker) -> None:
        self.__perf_marker = value

    @property
    def origin(self) -> str:
        return self.__origin

    @property
    def thread(self) -> str:
        return self.__thread

    @property
    def data(self) -> str:
        return self.__data

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.__timestamp < other.timestamp

    def __repr__(self) -> str:
        print(f"{self.id}\t{self.timestamp}\t{self.duration: 7.3f}\t{self.perf_marker}\t"
              f"{self.thread}\t{self.origin}\t{self.data}")


class EventStatistics:
    def __init__(self):
        self.__min_duration: float = 0
        self.__max_duration: float = 0
        self.__mean_duration: float = 0
        self.__med_duration: float = 0
        self.__durations_vector: List[float] = []

    def register_data(self, value: float) -> None:
        self.__min_duration = min(self.__max_duration, value)
        self.__max_duration = max(self.__max_duration, value)
        self.__durations_vector.append(value)

    def calculate_values(self) -> None:
        normalized_duration_vector = self.__durations_vector[1:]  # first duration is always 0
        self.__mean_duration = statistics.mean(normalized_duration_vector)
        self.__med_duration = statistics.median(normalized_duration_vector)

    @property
    def global_minimum(self) -> float:
        return self.__min_duration

    @property
    def global_maximum(self) -> float:
        return self.__min_duration

    @property
    def mean_value(self) -> float:
        return self.__mean_duration

    @property
    def median_value(self) -> float:
        return self.__med_duration

    @property
    def items_count(self) -> int:
        return len(self.__durations_vector)


class ThreadDescriptor:
    def __init__(self, thread_id: int, thread_name: str, stats: EventStatistics):
        if not thread_name or stats is None:
            raise ValueError("Thread name and statistics store reference must be not null")
        self.__thread_id = thread_id
        self.__events_vector: List[Event] = []
        self.__events_origin_set: Set[str] = set()
        self.__start_timestamp: [datetime, None] = None
        self.__end_timestamp: [datetime, None] = None
        self.__thread_name: str = thread_name
        self.__stats_store = stats

    @property
    def id(self):
        return self.__thread_id

    @property
    def start_timestamp(self) -> datetime:
        return self.__start_timestamp

    @property
    def end_timestamp(self) -> datetime:
        return self.__end_timestamp

    @property
    def duration(self) -> float:
        return (self.__end_timestamp - self.__start_timestamp).total_seconds()

    @property
    def thread_name(self) -> str:
        return self.__thread_name

    @property
    def origins_vector(self) -> List[str]:
        return [o for o in self.__events_origin_set]

    @property
    def size(self) -> int:
        return len(self.__events_vector)

    def calculate_durations(self) -> None:
        if not self.__stats_store:
            raise RuntimeError("TreadRepo wasn't properly initialized. Statistic store reference must be passed.")

        self.__events_vector.sort()
        timestamp_prev: [datetime, None] = None
        for index, event_item in enumerate(self.__events_vector):
            if timestamp_prev is None:
                timestamp_prev = event_item.timestamp
            else:
                duration_value = (event_item.timestamp - timestamp_prev).total_seconds()
                self.__events_vector[index].duration = duration_value
                self.__stats_store.register_data(duration_value)
                timestamp_prev = event_item.timestamp

    def add_event(self, event: Event) -> None:
        if self.__start_timestamp is None:
            self.__start_timestamp = event.timestamp
        else:
            self.__start_timestamp = min(self.__start_timestamp, event.timestamp)

        if self.__end_timestamp is None:
            self.__end_timestamp = event.timestamp
        else:
            self.__end_timestamp = max(self.__end_timestamp, event.timestamp)

        self.__events_vector.append(event)
        self.__events_origin_set.add(event.origin)

    def classify_items(self) -> None:
        mean_value: float = self.__stats_store.mean_value
        median_value: float = self.__stats_store.median_value

        for index, event_item in enumerate(self.__events_vector):
            duration_value = event_item.duration
            if duration_value <= 2 * median_value:
                self.__events_vector[index].perf_marker = PerformanceMarker.NORMAL
            else:
                if (duration_value > 2 * median_value) and (duration_value <= 1.5 * mean_value):
                    self.__events_vector[index].perf_marker = PerformanceMarker.WARNING
                else:
                    self.__events_vector[index].perf_marker = PerformanceMarker.CRITICAL

    def __iter__(self) -> Iterable[Event]:
        return self.__events_vector.__iter__()

    def __next__(self):
        return self.__events_vector.__iter__().__next__()

    def print_events(self):

        line_num = 1
        for event_item in self:
            print(f"{line_num: 4}\t{event_item.timestamp}\t{event_item.duration: 7.3f}\t{event_item.perf_marker.value}"
                  f"\t\t\t{event_item.origin}\t{event_item.data[:200]}")
            line_num += 1

    def __repr__(self):
        return f"TREAD #{self.id}: {self.thread_name.upper()} ▶ " \
               f"⏱ {self.duration:5.3f} s. ▶\tobjects: [{', '.join(self.origins_vector)}]\n"

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other: ThreadDescriptor):
        return self.start_timestamp < other.start_timestamp and self.end_timestamp < other.end_timestamp


class EventRepo:
    def __init__(self, log_filename):
        self.__event_store: Dict[str, ThreadDescriptor] = {}
        self.__stats_store = EventStatistics()
        self.__global_min_timestamp: [datetime, None] = None
        self.__global_max_timestamp: [datetime, None] = None
        self.__thread_count = 0
        self.__source_log_file = log_filename

    @property
    def log_filename(self):
        return self.__source_log_file

    @property
    def global_min_timestamp(self) -> datetime:
        return self.__global_min_timestamp

    @property
    def global_max_timestamp(self) -> datetime:
        return self.__global_max_timestamp

    @property
    def global_duration(self) -> float:
        return (self.global_max_timestamp - self.global_min_timestamp).total_seconds()

    @property
    def number_of_events(self) -> int:
        return self.__stats_store.items_count

    @property
    def number_of_treads(self) -> int:
        return self.__thread_count

    def register_event(self, event: Event) -> None:
        if event is None or not isinstance(event, Event):
            raise ValueError("Event reference must be not null & point to Event-class object")

        if self.__global_min_timestamp is None:
            self.__global_min_timestamp = event.timestamp
        else:
            self.__global_min_timestamp = min(self.__global_min_timestamp, event.timestamp)

        if self.__global_max_timestamp is None:
            self.__global_max_timestamp = event.timestamp
        else:
            self.__global_max_timestamp = min(self.__global_max_timestamp, event.timestamp)

        if event.thread not in self.__event_store:
            self.__thread_count += 1
            self.__event_store[event.thread] = ThreadDescriptor(self.__thread_count, event.thread, self.__stats_store)

        self.__event_store[event.thread].add_event(event)

    def process(self):
        for _, thread_store in self.__event_store.items():
            thread_store.calculate_durations()

        self.__stats_store.calculate_values()

        for _, thread_store in self.__event_store.items():
            thread_store.classify_items()

    def print_common_info(self):
        print("COMMON STATISTICS:\n")
        print(f"\tnumber of events - {self.number_of_events}")
        print(f"\ttotal duration\t - {self.global_duration:7.3f}")
        print(f"\t\t\tmin. - {self.__stats_store.global_minimum:7.3f}")
        print(f"\t\t\tmax. - {self.__stats_store.global_minimum:7.3f}")
        print(f"\t\t\tavg. - {self.__stats_store.mean_value:7.3f}")
        print(f"\t\t\tmed. - {self.__stats_store.median_value:7.3f}")
        print("\nTREADS SPECIFICATION:")
        for _, thread_content in self.__event_store.items():
            print(thread_content)

    def write_events_log(self, report_file: TextIO):
        report_file.write(f"CLASSIFIED EVENTS TABLE FROM ORIGINAL LOG-FILE '{self.__source_log_file}'\n")
        report_file.write("\nEvent classification symbols:\n")
        report_file.write("\t'+' (GOOD) - means, that duration of the event = [0, 2 x median value].\n")
        report_file.write("\t'-' (WARNING) - means, that duration of the event = (2 x median value, 1.5 x avg. value]\n")
        report_file.write("\t'!' (CRITICAL) - means, that duration of the event = (1.5 x avg. value, max. value)\n")
        report_file.write("\nReferential statistical values:\n")
        report_file.write(f"\tnumber of events - {self.number_of_events}\n")
        report_file.write(f"\ttotal duration\t - {self.global_duration:7.3f}\n")
        report_file.write(f"\t\t\tmin. - {self.__stats_store.global_minimum:7.3f}\n")
        report_file.write(f"\t\t\tmax. - {self.__stats_store.global_minimum:7.3f}\n")
        report_file.write(f"\t\t\tavg. - {self.__stats_store.mean_value:7.3f}\n")
        report_file.write(f"\t\t\tmed. - {self.__stats_store.median_value:7.3f}\n")
        report_file.write("\n   #\tTimestamp\t\t\t\t\t  Duration(sec)\t\tEvent origin (object)\t\t\t\t\tEvent data")
        report_file.write("--------------------------------------------------------------------------------------"
                          "--------------------------------------------------------------------------------------"
                          "------------------------\n")

        event_vector = []
        for thread_name, thread_content in self.__event_store.items():
            event_vector.extend(thread_content)
        event_vector.sort()
        report_file.writelines([e.__repr__() for e in event_vector])
        report_file.write("--------------------------------------------------------------------------------------"
                          "--------------------------------------------------------------------------------------"
                          "------------------------")


def extract_token(source: str, regex_pattern: Pattern[AnyStr], token_type: TokenType) -> \
        Union[str, datetime, EventType]:
    if not regex_pattern or not isinstance(regex_pattern, Pattern):
        raise ValueError(f"Incorrect pattern passed: {regex_pattern}")

    if not token_type or not isinstance(token_type, TokenType):
        raise ValueError(f"Incorrect token type passed: {token_type}")

    token_iter = re.finditer(regex_pattern, source)
    try:
        token_match_obj = token_iter.__next__()
        if token_match_obj is None or not token_match_obj.group(1):
            raise ValueError(f"Pattern '{regex_pattern}' returns an empty result from '{source}'")

        token_value_str = token_match_obj.group(1).strip()
        if token_type == TokenType.TIMESTAMP:
            try:
                return datetime.strptime(token_value_str, TIMESTAMP_FORMAT)
            except ValueError:
                raise ValueError(f"Incorrect timestamp format ('{TIMESTAMP_FORMAT}' expected) "
                                 f"at token '{token_value_str}' from source '{source}'")
        else:
            if token_type == TokenType.STRING:
                return token_value_str
            else:
                if token_type == TokenType.EVENT_TYPE:
                    try:
                        return EventType.from_string(token_value_str)
                    except ValueError:
                        raise ValueError(f"Incorrect event type format at token '{token_value_str}"
                                         f"' from source '{source}'")
                else:
                    raise ValueError(f"Unsupported token type: {token_type}")
    except StopIteration:
        raise ValueError(f"Pattern '{regex_pattern}' returns an empty result from '{source}'")
    except Exception:
        raise RuntimeError(f"Pattern '{regex_pattern}' raises unknown error during operation over the '{source}'")


if __name__ == '__main__':
    fileName = " ".join(sys.argv[1:])
    if not os.path.isfile(fileName):
        print(f"Can't open file: {fileName}")
        exit(1)

    f = open(file=fileName, mode="r")
    logContent = f.readlines()

    event_repo = EventRepo(fileName)

    for log_line in logContent:
        if len(log_line) == 0:
            continue

        try:
            event_timestamp_value = extract_token(log_line, TIMESTAMP_PATTERN, TokenType.TIMESTAMP)
        except ValueError:
            continue

        event_type_value = extract_token(log_line, EVENT_TYPE_PATTERN, TokenType.EVENT_TYPE)
        event_thread_value = extract_token(log_line, EVENT_THREAD_PATTERN, TokenType.STRING)
        event_origin_value = extract_token(log_line, EVENT_ORIGIN_PATTERN, TokenType.STRING)
        event_data_value = extract_token(log_line, EVENT_DATA_PATTERN, TokenType.STRING)

        event: Event = Event(event_timestamp_value,
                             event_type_value,
                             event_thread_value,
                             event_origin_value,
                             event_data_value)

        event_repo.register_event(event)

    event_repo.process()

    with open(f"Report on {os.path.basename(fileName)} ({datetime.now()}).txt", "w+") as output_file:
        event_repo.write_events_log(output_file)
