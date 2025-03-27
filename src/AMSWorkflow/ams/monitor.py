#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import json
import logging
import multiprocessing
import threading
import time
from collections import deque
from typing import Callable, List, Union


class AMSMonitor:
    """
    AMSMonitor can be used to decorate class methods and will
    record automatically the duration of the tasks in a hashmap
    with timestamp. The decorator will also automatically
    record the values of all attributes of the class.

        class ExampleTask1(Task):
            def __init__(self):
                self.total_bytes = 0
                self.total_bytes2 = 0

            # @AMSMonitor() would record all attributes
            # (total_bytes and total_bytes2) and the duration
            # of the block under the name amsmonitor_duration_ns.
            # Each time the same block (function in class or
            # predefined tag) is being monitored, AMSMonitor
            # create a new record with a timestamp (see below).
            #
            # @AMSMonitor(accumulate=True) records also all
            # attributes but does not create a new record each
            # time that block is being monitored, the first
            # timestamp is always being used and only
            # amsmonitor_duration_ns is being accumulated.
            # The user-managed attributes (like total_bytes
            # and total_bytes2 ) are not being accumulated.
            # By default, accumulate=False.

            # Example: we do not want to record total_bytes
            # but just total_bytes2
            @AMSMonitor(record=["total_bytes2"])
            def __call__(self):
                i = 0
                # Here we have to manually provide the current object being monitored
                with AMSMonitor(obj=self, tag="while_loop"):
                    while (i<=3):
                        self.total_bytes += 10
                        self.total_bytes2 = 1
                        i += 1

            # Example: We can also collect data at a finer grain
            @AMSMonitor(array=["myarray"])
            def f(self):
                i = 0
                while (i<2): myarray.append({"i":i})

    Each time `ExampleTask1()` is being called, AMSMonitor will
    populate `_stats` as follows (showed with two calls here):
        {
            "ExampleTask1": {
                "while_loop": {
                    "02/29/2024-19:27:53": {
                        "total_bytes2": 30,
                        "amsmonitor_duration_ns": 4.004607439041138
                    }
                },
                "__call__": {
                    "02/29/2024-19:29:24": {
                        "total_bytes2": 30,
                        "amsmonitor_duration_ns": 4.10461138
                    }
                },
                "f": {
                    "myarray": [
                        {
                            "i": 0,
                        },
                        {
                            "i": 1,
                        }
                }
            }
        }

    Attributes:
        record: attributes to record, if None, all attributes
            will be recorded, except objects (e.g., multiprocessing.Queue)
            which can cause problem. if empty ([]), no attributes will
            be recorded, only amsmonitor_duration_ns will be recorded.
        array: User can give a variable in which data can be accumulated over
            function calls. For example, `@AMSMonitor(array=["msg"])`
            give the possibilty to use the list `msg` within the decorated
            function to accumalate data.
        accumulate: If True, AMSMonitor will accumulate recorded
            data instead of recording a new timestamp for
            any subsequent call of AMSMonitor on the same method.
            We accumulate only records managed by AMSMonitor, like
            amsmonitor_duration_ns. We do not accumulate records
            from the monitored class/function.
        obj: Mandatory if using `with` statement, `object` is
            the main object should be provided (i.e., self).
        tag: Mandatory if using `with` statement, `tag` is the
            name that will appear in the record for that
            context manager statement.
    """

    _manager = multiprocessing.Manager()
    _stats = _manager.dict()
    _ts_format = "%m/%d/%Y-%H:%M:%S"
    _reserved_keys = ["amsmonitor_duration_ns"]
    _lock = threading.Lock()
    _count = 0

    def __init__(self, record=None, array=[], accumulate=False, obj=None, tag=None, logger: logging.Logger = None, **kwargs):
        self.accumulate = accumulate
        self.kwargs = kwargs
        self.record = record
        if not isinstance(record, list):
            self.record = None
        # We make sure we do not overwrite protected attributes managed by AMSMonitor
        if self.record:
            self.record = self._remove_reserved_keys(self.record)
        self.object = obj
        self.start_time = 0
        self.internal_ts = 0
        self.tag = tag
        AMSMonitor._count += 1
        self.logger = logger if logger else logging.getLogger(__name__)

        # Section to manage JSON array
        if not isinstance(array, list):
            array = [array]
        self.array_names = array
        self.variables_list = []
        self.array_context = {}

        for var in self.array_names:
            self.variables_list.append(deque())
            self.array_context[var] = self.variables_list[-1]

        # convenient bool to know if we need to support appending operations to array
        self.use_arrays = self.array_names != []

    def __str__(self) -> str:
        return AMSMonitor.info() if AMSMonitor._stats != {} else "{}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def lock():
        AMSMonitor._lock.acquire()

    @staticmethod
    def unlock():
        AMSMonitor._lock.release()

    def __enter__(self):
        if not self.object or not self.tag:
            self.logger.error('missing parameter "object" or "tag" when using context manager syntax')
            return
        self.start_monitor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitor()

    @classmethod
    def info(cls) -> str:
        s = ""
        if cls._stats == {}:
            return "{}"
        for class_name, func_calls in cls._stats.items():
            s += f"{class_name}\n"
            for func, categories in func_calls.items():
                s += f"  {func}\n"
                for cat_name, elems in categories.items():
                    s += f"    {cat_name:<10}\n"
                    for elem in elems:
                        for key, value in elem.items():
                            s += f"      {key:<30} => {value}\n"
                        s += f"\n"
        return s.rstrip()

    @classmethod
    @property
    def stats(cls):
        return cls._stats

    @classmethod
    @property
    def format_ts(cls):
        return cls._ts_format

    @classmethod
    def convert_ts(cls, ts: str) -> datetime.datetime:
        return datetime.strptime(ts, cls.format_ts)

    @classmethod
    def json(cls, json_output: str):
        """
        Write the collected metrics to a JSON file.
        """
        with open(json_output, "w") as fp:
            # we have to use .copy() as DictProxy is not serializable
            json.dump(cls._stats.copy(), fp, indent=4)
            # To avoid partial line at the end of the file
            fp.write("\n")

    @classmethod
    def reset(cls):
        cls.lock()
        cls._stats = cls._manager.dict()
        cls.unlock()

    def start_monitor(self, *args, **kwargs):
        self.start_time = time.time_ns()
        self.internal_ts = time.time_ns()

    def stop_monitor(self):
        end = time.time_ns()
        class_name = self.object.__class__.__name__
        func_name = self.tag

        new_data = vars(self.object)
        # Filter out multiprocessing which cannot be stored without causing RuntimeError
        new_data = self._filter_out_object(new_data)
        # We remove stuff we do not want (attribute of the calling class captured by vars())
        if self.record != []:
            new_data = self._filter(new_data, self.record)
        # We inject some data we want to record
        new_data["amsmonitor_duration_ns"] = end - self.start_time
        self._update_db(new_data, class_name, func_name, self.internal_ts)

        # We reinitialize some variables
        self.start_time = 0
        self.internal_ts = 0

    def __call__(self, func: Callable):
        """
        The main decorator.
        """

        def wrapper(*args, **kwargs):
            ts = time.time_ns()
            start = time.time_ns()

            if self.use_arrays:
                # Save copy of any global values that will be replaced.
                saved_values = {key: func.__globals__[key] for key in self.array_context if key in func.__globals__}
                func.__globals__.update(self.array_context)

            try:
                value = func(*args, **kwargs)
            finally:
                if self.use_arrays:
                    func.__globals__.update(saved_values)  # Restore any replaced globals.

            end = time.time_ns()
            if not hasattr(args[0], "__dict__"):
                return value
            class_name = args[0].__class__.__name__
            func_name = self.tag if self.tag else func.__name__
            if not self.use_arrays:
                # new_data is a dict of value from vars(). It contains all the class variable etc
                new_data = vars(args[0])
                # Filter out multiprocessing which cannot be stored without causing RuntimeError
                new_data = self._filter_out_object(new_data)

                # We remove stuff we do not want (attribute of the calling class captured by vars())
                new_data = self._filter(new_data, self.record)
                new_data["amsmonitor_duration_ns"] = (end - start)
            else:
                new_data = self.array_context
            self._update_db(new_data, class_name, func_name, ts)
            return value

        return wrapper

    def _update_db(self, new_data: dict, class_name: str, func_name: str, ts: str):
        """
        This function update the hashmap containing all the records.
        """
        if new_data == {}: return
        AMSMonitor.lock()
        if class_name not in AMSMonitor._stats:
            AMSMonitor._stats[class_name] = {}

        if func_name not in AMSMonitor._stats[class_name]:
            temp = AMSMonitor._stats[class_name]
            temp.update({func_name: {"records" : []}})
            AMSMonitor._stats[class_name] = temp
        temp = AMSMonitor._stats[class_name]


        # If we have to deal with arrays (if array != [])
        # Note that if we record arrays for this class / function
        # we do not record "records"
        if self.use_arrays:
            for tag in self.array_context:
                # Each tag has a list of elems
                while len(self.array_context[tag]) > 0:
                    # we remove the first elem to write it in the DB
                    elem = self.array_context[tag].popleft()
                    if tag not in temp[func_name]:
                        temp[func_name][tag] = []
                    temp[func_name][tag].append(elem)
        else:
            # We accumulate for each class with a different name
            if self.accumulate and temp[func_name] != []:
                ts = self._get_ts(class_name, func_name)
                temp[func_name]["records"][ts] = self._acc(temp[func_name][ts], new_data)
            else:
                item = {'timestamp': ts}
                for k, v in new_data.items():
                    item[k] = v
                temp[func_name]["records"].append(item)

        # This step is needed because AMSMonitor._stats is a manager.dict (not shared memory)
        # by reassigning the dictionary, the manager.dict is notified of the change
        AMSMonitor._stats[class_name] = temp
        AMSMonitor.unlock()
        # We flush the context array to receive the next chunk
        # self.array_context = []

    def _remove_reserved_keys(self, d: Union[dict, List]) -> dict:
        """
        Remove all the reserved keys from the dict given as input.
        """
        for key in self._reserved_keys:
            if key in d:
                self.logger.warning(f"attribute {key} is protected and will be ignored ({d})")
                if isinstance(d, list):
                    idx = d.index(key)
                    d.pop(idx)
                elif isinstance(d, dict):
                    del d[key]
        return d

    def _acc(self, original: dict, new_data: dict) -> dict:
        """
        Sum up element-wise two hashmaps (ignore fields that are not common)
        """
        for k, v in new_data.items():
            # We accumalate variable internally managed by AMSMonitor (duration etc)
            if k in AMSMonitor._reserved_keys:
                original[k] = float(original[k]) + float(v)
            else:
                original[k] = v
        return original

    def _filter_out_object(self, data: dict) -> dict:
        """
        Filter out a hashmap to remove objects which can cause errors
        """

        def is_serializable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False

        new_dict = {k: v for k, v in data.items() if is_serializable(v)}

        return new_dict

    def _filter(self, data: dict, keys: List[str]) -> dict:
        """
        Filter out a hashmap to contains only keys listed by list of keys
        """
        if not self.record:
            return data
        return {k: v for k, v in data.items() if k in keys}

    def _get_ts(self, class_name: str, func: str) -> int:
        """
        Return initial timestamp for a given monitored function.
        """
        ts = time.time_ns()
        if class_name not in AMSMonitor._stats or func not in AMSMonitor._stats[class_name]:
            return ts

        print(f"{class_name} {func} {AMSMonitor._stats}")
        init_ts = AMSMonitor._stats[class_name][func]
        if len(init_ts) == []:
            return ts
        if len(init_ts) > 1:
            self.logger.warning(f"more than 1 timestamp detected for {class_name} / {func}")
        return ts if init_ts == [] else init_ts[0]