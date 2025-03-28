#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import datetime
import json
import os
import time
import unittest

from ams.monitor import AMSMonitor
from ams.stage import Task


class ExampleTask1(Task):
    def __init__(self):
        self.x = 0
        self.y = 100

    @AMSMonitor()
    def __call__(self):
        i = 0
        with AMSMonitor(obj=self, record=["x"], tag="while_loop", accumulate=False):
            while 1:
                time.sleep(1)
                self.x += i
                if i == 3:
                    break
                i += 1
        self.y += 100

    @AMSMonitor(array=["myarray"])
    def f(self):
        i = 0
        while 1:
            myarray.append({"i":i})
            if i == 3:
                break
            i += 1

def read_json(path: str):
    with open(path) as f:
        d = json.load(f)
    return d


class TestMonitorTask1(unittest.TestCase):
    def setUp(self):
        self.task1 = ExampleTask1()

    def test_populating_monitor(self):
        AMSMonitor.reset()
        self.task1()
        self.task1.f()

        self.assertNotEqual(AMSMonitor.stats.copy(), {})
        self.assertIn("ExampleTask1", AMSMonitor.stats)
        self.assertIn("while_loop", AMSMonitor.stats["ExampleTask1"])
        self.assertIn("__call__", AMSMonitor.stats["ExampleTask1"])
        self.assertIn("f", AMSMonitor.stats["ExampleTask1"])

        for item in AMSMonitor.stats["ExampleTask1"]["__call__"]["records"]:
            self.assertIn("x", item)
            self.assertIn("y", item)
            self.assertIn("amsmonitor_duration_ns", item)
            self.assertEqual(item["x"], 6)
            self.assertEqual(item["y"], 200)

        for item in AMSMonitor.stats["ExampleTask1"]["while_loop"]["records"]:
            self.assertIn("x", item)
            self.assertNotIn("y", item)
            self.assertIn("amsmonitor_duration_ns", item)
            self.assertEqual(item["x"], 6)

        self.assertEqual(AMSMonitor.stats["ExampleTask1"]["f"]["records"], [])
        self.assertEqual(AMSMonitor.stats["ExampleTask1"]["f"]["myarray"], [{'i': 0}, {'i': 1}, {'i': 2}, {'i': 3}])

    def test_json_output(self):
        print(f"test_json_output {AMSMonitor.stats.copy()}")
        AMSMonitor.reset()
        self.task1()
        self.task1.f()
        path = "test_amsmonitor.json"
        AMSMonitor.json(path)
        self.assertTrue(os.path.isfile(path))
        d = read_json(path)
        self.assertEqual(AMSMonitor.stats.copy(), d)

    def tearDown(self):
        try:
            os.remove("test_amsmonitor.json")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
