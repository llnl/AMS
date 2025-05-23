#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import sys
import json
import os
import shutil
import signal
import time
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Optional, List
from enum import Enum
from multiprocessing import Process
from multiprocessing import Queue as mp_queue
from pathlib import Path
from queue import Queue as ser_queue
from threading import Thread

import numpy as np
from ams.faccessors import get_reader, get_writer
from ams.monitor import AMSMonitor
from ams.rmq import AMSMessage, AsyncConsumer, AMSRMQConfiguration, AsyncFanOutConsumer
from ams.store import AMSDataStore
from ams.util import get_unique_fn

BATCH_SIZE = 32 * 1024 * 1024


class MessageType(Enum):
    Process = 1
    NewModel = 2
    Terminate = 3
    Delete = 4


class DataBlob:
    """
    Class wrapping input, outputs in a single class

    Attributes:
        inputs: A ndarray of the inputs.
        outputs: A ndarray of the outputs.
    """

    def __init__(self, inputs, outputs, domain_name=None):
        self._domain_name = domain_name
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def domain_name(self):
        return self._domain_name


class QueueMessage:
    """
    A message in the IPC Queues.

    Attributes:
        msg_type: The type of the message. We currently support 3 types Process, NewModel, Terminate
        blob: The contents of the message
    """

    def __init__(self, msg_type, blob):
        if not isinstance(msg_type, MessageType):
            raise TypeError("Message Type should be of type MessageType")
        self.msg_type = msg_type
        self.blob = blob

    def is_terminate(self):
        return self.msg_type == MessageType.Terminate

    def is_process(self):
        return self.msg_type == MessageType.Process

    def is_delete(self):
        return self.msg_type == MessageType.Delete

    def is_new_model(self):
        return self.msg_type == MessageType.NewModel

    def data(self):
        return self.blob


class Task(ABC):
    """
    An abstract interface encapsulating a
    callable mechanism to be performed during
    the staging mechanism.
    """

    @abstractmethod
    def __call__(self):
        pass


class ForwardTask(Task):
    """
    A ForwardTask reads messages from some input queues performs some
    action/transformation and forwards the outcome to some output queue.

    Attributes:
        application_name: The name of the application currently being used
        db_url: url to sql server that stores metadata of files.
        i_queue: The input queue to read input message
        o_queue: The output queue to write the transformed messages
        user_obj: An object providing the update_model_cb and data_cb callbacks to be applied on the respective control messages before pushing it to the next stage.
    """

    def __init__(self, application_name, db_url, i_queue, o_queue, user_obj):
        """
        initializes a ForwardTask class with the queues and the callback.
        """
        self.application_name = application_name
        self.db_url = db_url
        self.i_queue = i_queue
        self.o_queue = o_queue
        self.user_obj = user_obj
        self.datasize_byte = 0

    def _data_cb(self, data):
        """
        Apply an 'action' to the incoming data

        Args:
            data: A DataBlob of inputs, outputs to be transformed

        Returns:
            A pair of inputs, outputs of the data after the transformation
        """
        inputs, outputs = self.user_obj.data_cb(data.inputs, data.outputs)
        # This can be too conservative, we may want to relax it later
        if not (isinstance(inputs, np.ndarray) and isinstance(outputs, np.ndarray)):
            raise TypeError(
                f"{self.user_obj.__name__}.data_cb did not return numpy arrays"
            )
        return inputs, outputs

    def _model_update_cb(self, db, msg):
        domain = msg["domain"]
        model = db.search(domain, "models", version="latest")
        _updated = self.user_obj.update_model_cb(domain, model)
        print(f"Model update status: {_updated}")

    @AMSMonitor(record=["datasize_byte"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue, acting on those messages and forwarding
        the output to the output queue. In the case of receiving a 'termination' messages informs
        the tasks waiting on the output queues about the terminations and returns from the function.
        """

        with AMSDataStore(self.application_name, self.db_url) as db:
            while True:
                # This is a blocking call
                item = self.i_queue.get(block=True)
                if item.is_terminate():
                    print(f"Received Terminate {self.__class__.__name__}")
                    self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                    break
                elif item.is_process():
                    data = item.data()
                    inputs, outputs = self._data_cb(data)
                    self.o_queue.put(
                        QueueMessage(
                            MessageType.Process,
                            DataBlob(inputs, outputs, data.domain_name),
                        )
                    )
                    self.datasize_byte += inputs.nbytes + outputs.nbytes
                elif item.is_new_model():
                    data = item.data()
                    self._model_update_cb(db, data)
                elif item.is_delete():
                    print(f"Sending Delete Message Type {self.__class__.__name__}")
                    self.o_queue.put(item)
        return


class FSLoaderTask(Task):
    """
    A FSLoaderTask reads files from the filesystem bundles the data of
    the files into batches and forwards them to the next task waiting on the
    output queuee.

    Attributes:
        o_queue: The output queue to write the transformed messages
        loader: A child class inheriting from FileReader that loads data from the filesystem.
        pattern: The (glob-)pattern of the files to be read.
    """

    def __init__(self, o_queue, loader, pattern):
        self.o_queue = o_queue
        self.pattern = pattern
        self.loader = loader
        self.datasize_byte = 0
        self.total_time_ns = 0

    @AMSMonitor(array=["msgs"], record=["datasize_byte", "total_time_ns"])
    def __call__(self):
        """
        Busy loop of reading all files matching the pattern and creating
        '100' batches which will be pushed on the queue. Upon reading all files
        the Task pushes a 'Terminate' message to the queue and returns.
        """

        start = time.time_ns()
        files = list(glob.glob(self.pattern))
        for fn in files:
            start_time_fs = time.time_ns()
            with self.loader(fn) as fd:
                domain_name, input_data, output_data = fd.load()
                print("Domain Name is", domain_name)
                row_size = input_data[0, :].nbytes + output_data[0, :].nbytes
                rows_per_batch = int(np.ceil(BATCH_SIZE / row_size))
                num_batches = int(np.ceil(input_data.shape[0] / rows_per_batch))
                input_batches = np.array_split(input_data, num_batches)
                output_batches = np.array_split(output_data, num_batches)
                for j, (i, o) in enumerate(zip(input_batches, output_batches)):
                    self.o_queue.put(
                        QueueMessage(MessageType.Process, DataBlob(i, o, domain_name))
                    )
                self.datasize_byte += input_data.nbytes + output_data.nbytes

                end_time_fs = time.time_ns()
                msg = {
                    "file": fn,
                    "domain_name": domain_name,
                    "row_size": row_size,
                    "batch_size": BATCH_SIZE,
                    "rows_per_batch": rows_per_batch,
                    "num_batches": num_batches,
                    "size_bytes": input_data.nbytes + output_data.nbytes,
                    "process_time_ns": end_time_fs - start_time_fs,
                }
                # msgs is a list that is managed by AMSMonitor, we simply append to it
                msgs.append(msg)

            print(f"Sending Delete Message Type {self.__class__.__name__}")
            self.o_queue.put(QueueMessage(MessageType.Delete, fn))
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))

        end = time.time_ns()
        self.total_time_ns += end - start
        print(f"Spend {(end - start)/1e9} at {self.__class__.__name__}")


class RMQDomainDataLoaderTask(Task):
    """
    A RMQDomainDataLoaderTask consumes 'AMSMessages' from RabbitMQ bundles the data of
    the files into batches and forwards them to the next task waiting on the
    output queuee.

    Attributes:
        o_queue: The output queue to write the transformed messages
        credentials: A JSON file with the credentials to log on the RabbitMQ server.
        certificates: TLS certificates
        rmq_queue: The RabbitMQ queue to listen to.
        prefetch_count: Number of messages prefected by RMQ (impact performance)
    """

    def __init__(
        self,
        o_queue,
        host,
        port,
        vhost,
        user,
        password,
        cert,
        rmq_queue,
        policy,
        prefetch_count=1,
        signals=[signal.SIGINT, signal.SIGUSR1],
    ):
        self.o_queue = o_queue
        self.cert = cert
        self.rmq_queue = rmq_queue
        self.prefetch_count = prefetch_count
        self.datasize_byte = 0
        self.total_time_ns = 0
        self.signals = signals
        self.orig_sig_handlers = {}
        self.policy = policy

        # Signals can only be used within the main thread
        if self.policy != "thread":
            # We ignore SIGTERM, SIGUSR1, SIGINT by default so later
            # we can override that handler only for RMQDomainDataLoaderTask
            for s in self.signals:
                self.orig_sig_handlers[s] = signal.getsignal(s)
                signal.signal(s, signal.SIG_IGN)

        self.rmq_consumer = AsyncConsumer(
            host=host,
            port=port,
            vhost=vhost,
            user=user,
            password=password,
            cert=self.cert,
            queue=self.rmq_queue,
            on_message_cb=self.callback_message,
            on_close_cb=self.callback_close,
            prefetch_count=self.prefetch_count,
        )

    def callback_close(self):
        """
        Callback that will be called when RabbitMQ will close
        the connection (or if a problem happened with the connection).
        """
        print("Adding terminate message at queue:", self.o_queue)
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))

    @AMSMonitor(array=["msgs"], record=["datasize_byte", "total_time_ns"])
    def callback_message(self, ch, basic_deliver, properties, body):
        """
        Callback that will be called each time a message will be consummed.
        the connection (or if a problem happened with the connection).
        """
        start_time = time.time_ns()
        msg = AMSMessage(body)
        domain_name, input_data, output_data = msg.decode()
        self.datasize_byte += input_data.nbytes + output_data.nbytes

        self.o_queue.put(
            QueueMessage(
                MessageType.Process, DataBlob(input_data, output_data, domain_name)
            )
        )
        end_time = time.time_ns()
        self.total_time_ns += end_time - start_time
        # TODO: Improve the code to manage potentially multiple messages per AMSMessage
        msg = {
            "delivery_tag": basic_deliver.delivery_tag,
            "mpi_rank": msg.mpi_rank,
            "domain_name": domain_name,
            "num_elements": input_data.size + output_data.size,
            "input_dim": msg.input_dim,
            "output_dim": msg.output_dim,
            "size_bytes": input_data.nbytes + output_data.nbytes,
            "ts_received": start_time,
            "ts_processed": end_time,
        }
        msgs.append(msg)

    def signal_wrapper(self, name, pid):
        def handler(signum, frame):
            print(f"Received SIGNUM={signum} for {name}[pid={pid}]: stopping process")
            self.stop()

        return handler

    def stop(self):
        print(f"Stopping {self.__class__.__name__}")
        self.rmq_consumer.stop()
        print(f"Spend {self.total_time_ns/1e9} at {self.__class__.__name__}")

    def __call__(self):
        """
        Busy loop of consuming messages from RMQ queue
        """
        # Installing signal callbacks only for RMQDomainDataLoaderTask
        if self.policy != "thread":
            for s in self.signals:
                signal.signal(
                    s, self.signal_wrapper(self.__class__.__name__, os.getpid())
                )
        print(f"{self.__class__.__name__} PID is:", os.getpid())
        self.rmq_consumer.run()
        print(f"Returning from {self.__class__.__name__}")


class RMQControlMessageTask(RMQDomainDataLoaderTask):
    """
    A RMQControlMessageTask consumes JSON-messages from RabbitMQ and forwards them to
    the o_queue of the pruning Task.

    Attributes:
        o_queue: The output queue to write the transformed messages
        credentials: A JSON file with the credentials to log on the RabbitMQ server.
        certificates: TLS certificates
        rmq_queue: The RabbitMQ queue to listen to.
        prefetch_count: Number of messages prefected by RMQ (impact performance)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def callback_message(self, ch, basic_deliver, properties, body):
        """
        Callback that will be called each time a message will be consummed.
        the connection (or if a problem happened with the connection).
        """
        start_time = time.time()
        data = json.loads(body)
        if data["request_type"] == "done-training":
            self.o_queue.put(QueueMessage(MessageType.NewModel, data))

        self.total_time_ns += time.time_ns() - start_time


class AMSShutdown(AsyncFanOutConsumer):
    """
    A RMQ consumer client that listens to control messages from the AMS Deployment tool. When it receives a 'terminate' message
    it shutdown the rest of the connections/threads to the RMQ server and gracefully terminates.
    """

    def __init__(
        self,
        consumers: List[RMQDomainDataLoaderTask],
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        prefetch_count: int = 1,
    ):
        self._consumers = consumers
        super().__init__(
            host,
            port,
            vhost,
            user,
            password,
            cert,
            "",
            prefetch_count,
            on_message_cb=self.on_message_cb,
            on_close_cb=self.on_close_cb,
        )
        self._signal_pid = []

    def on_message_cb(self, ch, basic_deliver, properties, body):
        message = json.loads(body)
        print(f"Received Signalling {message}")
        if "request_type" in message:
            if message["request_type"] == "terminate":
                # NOTE: when my pids are empty, I consider that I am running through threading,
                # and thus I have access to the class. Otherwise, I am executing through
                # multiprocessing and I need to "kill" the application.
                if len(self._signal_pid) == 0:
                    for consumer in self._consumers:
                        consumer.stop()
                else:
                    for sig_num in self._signal_pid:
                        os.kill(sig_num, signal.SIGINT)
                self.stop()

    def on_close_cb(self):
        print("Closing")

    def __call__(self, pids=None):
        if pids:
            print("Received PIDS: ", pids)
            self._signal_pid = pids
        self.run()


class FSWriteTask(Task):
    """
    A Class representing a task flushing data in the specified output directory

    Attributes:
        i_queue: The input queue to read data from.
        o_queue: The output queue to write the path of the saved file.
        writer_cls: A child class inheriting from FileWriter that writes to the specified file.
        out_dir: The directory to write data to.
    """

    def __init__(self, i_queue, o_queue, writer_cls, out_dir):
        """
        initializes the writer task to read data from the i_queue write them using
        the writer_cls and store the data in the out_dir.
        """
        self.data_writer_cls = writer_cls
        self.out_dir = out_dir
        self.i_queue = i_queue
        self.o_queue = o_queue
        self.suffix = writer_cls.get_file_format_suffix()

        # Max size in byte before writing a new file
        self.max_size_file = os.getenv("AMS_MAX_FILE_SIZE", 2*1024*1024*1024)
        # We print something everything X messages processed
        self.print_message = os.getenv("AMS_MAX_PRINT_MESSAGE", 1000)

    @AMSMonitor(array=["requests"])
    def process_request(self, data_files, item):
        """
        Function that process a request for FSwriteTask and write the file on disk
        """
        start_time_req = time.time_ns()
        data = item.data()
        if data.domain_name not in data_files:
            fn = get_unique_fn()
            fn = f"{self.out_dir}/{data.domain_name}_{fn}.{self.suffix}"
            # TODO: bytes_written should be an attribute of the file
            # to keep track of the size of the current file. Currently we keep track of this
            # by keeping a value in a list
            data_files[data.domain_name] = [
                self.data_writer_cls(fn).open(),
                0,
            ]
        bytes_written = data.inputs.size * data.inputs.itemsize
        bytes_written += data.outputs.size * data.outputs.itemsize

        data_files[data.domain_name][0].store(data.inputs, data.outputs)
        data_files[data.domain_name][1] += bytes_written

        self.total_bytes_written += data.inputs.size * data.inputs.itemsize
        self.total_bytes_written += data.outputs.size * data.outputs.itemsize

        if data_files[data.domain_name][1] >= self.max_size_file:
            data_files[data.domain_name][0].close()
            self.o_queue.put(
                QueueMessage(
                    MessageType.Process,
                    (
                        data.domain_name,
                        data_files[data.domain_name][0].file_name,
                    ),
                )
            )
            del data_files[data.domain_name]

        end_time_req = time.time_ns()
        req = {
            "request_id": self.total_messages,
            "domain_name": data.domain_name,
            "file_size": bytes_written,
            "total_bytes_written": self.total_bytes_written,
            "max_size_file": self.max_size_file,
            "timestamp": start_time_req,
            "process_time_ns": end_time_req - start_time_req,
        }
        requests.append(req)

    @AMSMonitor(record=["datasize_byte"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue, writting the input,output data in a file
        using the instances 'writer_cls' and inform the task waiting on the output_q about the
        path of the file.
        """

        start = time.time()
        self.total_bytes_written = 0
        data_files = dict()
        self.total_messages = 0
        with AMSMonitor(obj=self, tag="internal_loop", accumulate=False):
            while True:
                # This is a blocking call
                item = self.i_queue.get(block=True)
                self.total_messages += 1
                if item.is_terminate():
                    for k, v in data_files.items():
                        v[0].close()
                        self.o_queue.put(
                            QueueMessage(MessageType.Process, (k, v[0].file_name))
                        )
                    del data_files
                    self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                    break
                elif item.is_delete():
                    print(f"Sending Delete Message Type {self.__class__.__name__}")
                    self.o_queue.put(item)
                elif item.is_process():
                    self.process_request(data_files, item)

                if self.total_messages % self.print_message == 0:
                    print(
                        f"I have processed {self.total_messages} in total amounting to {self.total_bytes_written/(1024.0*1024.0)} MB"
                    )

        end = time.time()
        self.datasize_byte = self.total_bytes_written
        print(f"Spend {end - start} {self.total_bytes_written} at {self.__class__.__name__}")


class PushToStore(Task):
    """
    PushToStore is the epilogue of the pipeline. Effectively (if instructed so) it informs the kosh store
    about the existence of a new file.

    Attributes:
        ams_config: The AMS configuration storing information regarding the AMS setup.
        i_queue: The queue to read file locations from
        dir: The directory of the database
        store: The Kosh Store
    """

    def __init__(
        self,
        i_queue,
        application_name,
        dest_dir,
        db_url=None,
    ):
        """
        Initializes the PushToStore Task.
        Args:
            i_queue: The queue to read requests from.
            application_name: The name of the running application.
            dest_dir: The path to store persistend data to.
            db_url: The url to a SQL DB server which will be used to register metadata associated with the files.
        """

        self.i_queue = i_queue
        self.dest_dir = Path(dest_dir).absolute()
        self.db_url = db_url
        self.nb_requests = 0
        self.application_name = application_name
        self.total_filesize = 0
        if not self.dest_dir.exists():
            self.dest_dir.mkdir(parents=True, exist_ok=True)

    @AMSMonitor(record=["nb_requests", "total_filesize"], array=["requests"])
    def process_request(self, db_store, item):
        """
        Function that process a request to push the data to the DB
        """
        start_time_req = time.time_ns()
        self.nb_requests += 1
        domain_name, file_name = item.data()
        if domain_name == None:
            domain_name = "unknown-domain"
        src_fn = Path(file_name)
        dest_file = self.dest_dir / src_fn.name
        if src_fn != dest_file:
            shutil.move(src_fn, dest_file)
        if self.db_url is not None:
            db_store.add_candidates(domain_name, [str(dest_file)])

        self.total_filesize += os.stat(src_fn).st_size
        end_time_req = time.time_ns()

        req = {
            "request_id": self.nb_requests,
            "domain_name": domain_name,
            "file_name": file_name,
            "file_size": os.stat(src_fn).st_size,
            "total_size": self.total_filesize,
            "timestamp": start_time_req,
            "process_time_ns": end_time_req - start_time_req,
        }
        requests.append(req)

    @AMSMonitor(record=["nb_requests"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue publishing them to the underlying store.
        """
        start = time.time()
        if self.db_url is not None:
            db_store = AMSDataStore(self.application_name, self.db_url).open()

        with AMSMonitor(obj=self, tag="internal_loop", record=["nb_requests", "application_name", "total_filesize"]):
            while True:
                item = self.i_queue.get(block=True)
                if item.is_terminate():
                    print(f"Received Terminate {self.__class__.__name__}")
                    break
                if item.is_delete():
                    print(f"Sending Delete Message Type {self.__class__.__name__}")
                    fn = item.data()
                    Path(fn).unlink()
                elif item.is_process():
                    print(f"Got message to store in DB {item}")
                    self.process_request(db_store, item)

        end = time.time()
        print(f"Spend {end - start} at {self.__class__.__name__}")


class Pipeline(ABC):
    """
    An interface class representing a sequence of transformations/actions to be performed
    to store data in the AMS kosh-store. The actions can be performed either sequentially,
    or in parallel using different poclies/vehicles (threads or processes).

    Attributes:
        ams_config: The AMS configuration required when publishing to the AMS store.
        dest_dir: The final path to store data to.
        stage_dir: An intermediate location to store files. Usefull if the configuration requires
            storing the data in some scratch directory (SSD) before making them public to the parallel filesystem.
        actions: A list of actions to be performed before storing the data in the filesystem
        db_type: The file format of the data to be stored
        writer: The class to be used to write data to the filesystem.
    """

    supported_policies = {"sequential", "thread", "process"}
    supported_writers = {"shdf5", "dhdf5"}

    def __init__(self, application_name, dest_dir, db_url, db_type="dhdf5"):
        """
        initializes the Pipeline class to write the final data in the 'dest_dir' using a file writer of type 'db_type'
        and optionally caching the data in the 'stage_dir' before making them available in the cache store.
        """

        self.application_name = application_name
        self.dest_dir = dest_dir
        self.user_action = None
        self.db_type = db_type
        self._writer = get_writer(self.db_type)
        self.db_url = db_url

        # For signal handling
        self.released = False

        self.signals = [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1]

    def signal_wrapper(self, name, pid):
        def handler(signum, frame):
            print(f"Received SIGNUM={signum} for {name}[pid={pid}]")
            # We trigger the underlying signal handlers for all tasks
            # This should only trigger RMQDomainDataLoaderTask

            # TODO: I don't like this system to shutdown the pipeline on demand
            # It's extremely easy to mess thing up with signals.. and it's
            # not a robust solution (if a task is not managing correctly SIGINT
            # the pipeline can explode)
            for e in self._executors:
                os.kill(e.pid, signal.SIGINT)
            self.release_signals()

        return handler

    def init_signals(self):
        self.released = False
        self.original_handlers = {}
        for sig in self.signals:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(
                sig, self.signal_wrapper(self.__class__.__name__, os.getpid())
            )

    def release_signals(self):
        if not self.released:
            # We put back all the signal handlers
            for sig in self.signals:
                signal.signal(sig, self.original_handlers[sig])

            self.released = True

    def add_user_action(self, obj):
        """
        Adds an action to be performed at the data before storing them in the filesystem

        Args:
            callback: A callback to be called on every input, output.
        """

        if not (hasattr(obj, "data_cb") and callable(getattr(obj, "data_cb"))):
            raise TypeError(f"User provided object {obj} does not have data_cb")

        if not (
            hasattr(obj, "update_model_cb")
            and callable(getattr(obj, "update_model_cb"))
        ):
            raise TypeError(f"User provided object {obj} does not have data_cb")

        self.user_action = obj

    def _seq_execute(self):
        """
        Executes all tasks sequentially. Every task starts after all incoming messages
        are processed by the previous task.
        """
        for t in self._tasks:
            t()

    def _parallel_execute(self, exec_vehicle_cls):
        """
        parallel execute of all tasks using the specified vehicle type

        Args:
            exec_vehicle_cls: The class to be used to generate entities
            executing actions by reading data from i/o_queue(s).
        """
        self._executors = list()
        for a in self._tasks:
            self._executors.append(exec_vehicle_cls(target=a))

        for e in self._executors:
            e.start()

        pids_to_kill = []
        if isinstance(self._tasks[0], RMQDomainDataLoaderTask):
            print("My task is the right one, I need to kill it")
            pids_to_kill.append(self._executors[0].pid)
        print("Pids to kill are", pids_to_kill)
        shutdown_task = exec_vehicle_cls(target=self.shutdown, args=([pids_to_kill]))
        shutdown_task.start()

        shutdown_task.join()
        for e, t in zip(self._executors, self._tasks):
            e.join()

    def _execute_tasks(self, policy):
        """
        Executes all tasks using the specified policy

        Args:
            policy: The policy to be used to execute the pipeline
        """
        executors = {"thread": Thread, "process": Process}

        if policy == "sequential":
            self._seq_execute()
            return

        self._parallel_execute(executors[policy])
        return

    def _link_pipeline(self, policy):
        """
        Links all actions/stages of the pipeline with input/output queues.

        Args:
            policy: The policy to be used to execute the pipeline
        """
        _qType = self.get_q_type(policy)
        # We need 1 queue to copy incoming data to the pipeline
        # Every action requires 1 input and one output q. But the output
        # q is used as an inut q on the next action thus we need num actions -1.
        # 2 extra queues to store to data-store and publish on kosh
        self._queues = [_qType() for i in range(3)]

        self._tasks = [self.get_load_task(self._queues[0], policy)]

        self._tasks.append(
            ForwardTask(
                self.application_name,
                self.db_url,
                self._queues[0],
                self._queues[1],
                self.user_action,
            )
        )
        if self.requires_model_update():
            self._tasks.append(self.get_model_update_task(self._queues[0], policy))

        # After user actions we store into a file
        self._tasks.append(
            FSWriteTask(self._queues[1], self._queues[2], self._writer, self.dest_dir)
        )
        # After storing the file we make it public to the store.
        self._tasks.append(
            PushToStore(
                self._queues[2], self.application_name, self.dest_dir, self.db_url
            )
        )

    def execute(self, policy):
        """
        Execute the pipeline of tasks using the specified policy (blocking).

        Args:
            policy: The policy to be used to execute the pipeline
        """
        if policy not in self.__class__.supported_policies:
            raise RuntimeError(
                f"Pipeline execute does not support policy: {policy}, please select from  {Pipeline.supported_policies}"
            )

        self.init_signals()
        # Create a pipeline of actions and link them with appropriate queues
        self._link_pipeline(policy)
        # Execute them
        self._execute_tasks(policy)
        self.release_signals()

    @abstractmethod
    def requires_model_update(self):
        """
        Returns whether the pipeline provides a model-update message parsing mechanism
        """
        return False

    @abstractmethod
    def get_model_update_task(self, o_queue, policy):
        pass

    @abstractmethod
    def get_load_task(self, o_queue, policy):
        """
        Callback to the child class to return the task that loads data from some unspecified entry-point.
        """
        pass

    @staticmethod
    @abstractmethod
    def add_cli_args(parser):
        """
        Initialize root pipeline class cli parser with the options.
        """
        parser.add_argument(
            "--dest",
            "-d",
            dest="dest_dir",
            help="Where to store the data (Directory should exist)",
        )
        parser.add_argument(
            "--db-url",
            "-url",
            dest="db_url",
            help="The SQL url to store the metadata to",
            required=True,
        )
        parser.add_argument(
            "--application-name",
            "-a",
            dest="application_name",
            help="The name of the application we will store data and metadata for",
            required=True,
        )
        parser.add_argument(
            "--db-type",
            dest="db_type",
            choices=Pipeline.supported_writers,
            help="File format to store the data to",
            default="dhdf5",
        )
        return

    @abstractmethod
    def from_cli(cls):
        pass

    @staticmethod
    def get_q_type(policy):
        """
        Returns the type of the queue to be used to create Queues for the specified policy.
        """

        p_to_type = {"sequential": ser_queue, "thread": ser_queue, "process": mp_queue}
        return p_to_type[policy]

    @abstractmethod
    def shutdown(self):
        pass


class FSPipeline(Pipeline):
    """
    A 'Pipeline' reading data from the Filesystem and storing them back to the filesystem.

    Attributes:
        src: The source directory to read data from.
        pattern: The pattern to glob files from.
        src_type: The file format of the source data
    """

    supported_readers = ("shdf5", "dhdf5")

    def __init__(
        self, application_name, dest_dir, db_url, db_type, src, src_type, pattern
    ):
        """
        Initialize a FSPipeline that will write data to the 'dest_dir' and optionally publish
        these files to the kosh-store 'store' by using the stage_dir as an intermediate directory.
        """
        super().__init__(application_name, dest_dir, db_url, db_type)
        self._src = Path(src)
        self._pattern = pattern
        self._src_type = src_type

    def get_load_task(self, o_queue, policy):
        """
        Return a Task that loads data from the filesystem

        Args:
            o_queue: The queue the load task will push read data.

        Returns: An FSLoaderTask instance reading data from the filesystem and forwarding the values to the o_queue.
        """
        loader = get_reader(self._src_type)
        return FSLoaderTask(
            o_queue, loader, pattern=str(self._src) + "/" + self._pattern
        )

    @staticmethod
    def add_cli_args(parser):
        """
        Add cli arguments to the parser required by this Pipeline.
        """
        Pipeline.add_cli_args(parser)
        parser.add_argument(
            "--src", "-s", help="Where to copy the data from", required=True
        )
        parser.add_argument(
            "--src-type", "-st", choices=FSPipeline.supported_readers, default="shdf5"
        )
        parser.add_argument(
            "--pattern", "-p", help="Glob pattern to read data from", required=True
        )
        return

    @classmethod
    def from_cli(cls, args):
        """
        Create FSPipeline from the user provided CLI.
        """
        return cls(
            args.application_name,
            args.dest_dir,
            args.db_url,
            args.db_type,
            args.src,
            args.src_type,
            args.pattern,
        )

    def requires_model_update(self):
        return False

    def get_model_update_task(self, o_queue, policy):
        raise RuntimeError("FSPipeline does not support model update")

    def shutdown(self, pids):
        pass


class RMQPipeline(Pipeline):
    """
    A 'Pipeline' reading data from RabbitMQ and storing them back to the filesystem.

    Attributes:
        host: RabbitMQ host
        port: RabbitMQ port
        vhost: RabbitMQ virtual host
        user: RabbitMQ username
        password: RabbitMQ password for username
        cert: The TLS certificate
        rmq_queue: The RMQ queue to listen to.
    """

    def __init__(
        self,
        application_name,
        dest_dir,
        db_url,
        db_type,
        host,
        port,
        vhost,
        user,
        password,
        cert,
        data_queue,
        model_update_queue=None,
    ):
        """
        Initialize a RMQPipeline that will write data to the 'dest_dir' and optionally publish
        these files to the kosh-store 'store' by using the stage_dir as an intermediate directory.
        """
        super().__init__(application_name, dest_dir, db_url, db_type)
        self._host = host
        self._port = port
        self._vhost = vhost
        self._user = user
        self._password = password
        self._cert = Path(cert)
        self._data_queue = data_queue
        self._model_update_queue = model_update_queue
        print("Received a data queue of", self._data_queue)
        print("Received a model_update queue of", self._model_update_queue)
        self._gracefull_shutdown = None
        self._o_queue = None

    def get_load_task(self, o_queue, policy):
        """
        Return a Task that loads data from the filesystem

        Args:
            o_queue: The queue the load task will push read data.

        Returns: An RMQDomainDataLoaderTask instance reading data from the
        filesystem and forwarding the values to the o_queue.
        """

        Loader = RMQDomainDataLoaderTask(
            o_queue,
            self._host,
            self._port,
            self._vhost,
            self._user,
            self._password,
            self._cert,
            self._data_queue,
            policy,
            prefetch_count=1,
        )
        self._o_queue = o_queue
        self._gracefull_shutdown = AMSShutdown(
            [Loader],
            self._host,
            self._port,
            self._vhost,
            self._user,
            self._password,
            self._cert,
        )

        return Loader

    def get_model_update_task(self, o_queue, policy):
        """
        Return a Task receives messages from the training job regarding the status of new models

        Args:
            o_queue: The queue to push the model update message.

        Returns: An RMQControlMessageTask instance reading data from self._model_update_queue
        and forwarding the values to the o_queue.
        """

        # The model update tasks does not need to have a gracefull shutdown.
        # TODO: We need to think about this once we actually use it
        return RMQControlMessageTask(
            o_queue,
            self._host,
            self._port,
            self._vhost,
            self._user,
            self._password,
            self._cert,
            self._model_update_queue,
            policy,
        )

    @staticmethod
    def add_cli_args(parser):
        """
        Add cli arguments to the parser required by this Pipelinereturn .
        """
        Pipeline.add_cli_args(parser)
        parser.add_argument(
            "-c", "--creds", help="AMS credentials file (JSON)", required=True
        )
        parser.add_argument(
            "-u", "--update-rmq-models", help="Update-rmq-models", action="store_true"
        )
        return

    @classmethod
    def from_cli(cls, args):
        """
        Create RMQPipeline from the user provided CLI.
        """

        config = AMSRMQConfiguration.from_json(args.creds)

        return cls(
            args.application_name,
            args.dest_dir,
            args.db_url,
            args.db_type,
            config.service_host,
            config.service_port,
            config.rabbitmq_vhost,
            config.rabbitmq_user,
            config.rabbitmq_password,
            config.rabbitmq_cert,
            config.rabbitmq_queue_physics,
            config.rabbitmq_exchange_training if args.update_rmq_models else None,
        )

    def requires_model_update(self):
        return self._model_update_queue is not None

    def shutdown(self, pid):
        print(f"Waiting in shutdown {self.__class__.__name__}")
        self._gracefull_shutdown(pid)
        print(f"Received Terminate {self.__class__.__name__}")
        self._o_queue.put(QueueMessage(MessageType.Terminate, None))


def get_pipeline(src_mechanism="fs"):
    """
    Factory method to return the pipeline mechanism for the given source entry point

    Args:
        src_mechanism: The entry mechanism to read data from.

    Returns: A Pipeline class to start the stage AMS service
    """
    pipe_mechanisms = {"fs": FSPipeline, "network": RMQPipeline}
    if src_mechanism not in pipe_mechanisms.keys():
        raise RuntimeError(f"Pipeline {src_mechanism} storing mechanism does not exist")
    return pipe_mechanisms[src_mechanism]
