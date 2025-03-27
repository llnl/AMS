from ams.ams_jobs import (
    AMSDomainJob,
    AMSNetworkStageJob,
    AMSMLTrainJob,
    AMSOrchestratorJob,
    AMSSubSelectJob,
)
import os
import time
from ams.ams_flux import AMSFluxExecutor
from ams.ams_jobs import AMSJob, AMSJobResources
from ams.rmq import (
    AMSFanOutProducer,
    AMSRMQConfiguration,
    AMSSyncProducer,
    StatusPoller,
)
from ams.store import AMSDataStore, create_store_directories
import flux
import json
from flux.job.list import get_job
from concurrent.futures import wait

from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import threading


def get_allocation_resources(uri: str) -> Tuple[int, int, int]:
    """
    @brief Returns the resources of a flux allocation

    :param uri: A flux uri to querry the resources from
    :return: A tuple of (nnodes, cores_per_node, gpus_per_node)
    """
    flux_instance = flux.Flux(uri)
    resources = flux.resource.resource_list(flux_instance).get()["all"]
    cores_per_node = int(resources.ncores / resources.nnodes)
    gpus_per_node = int(resources.ngpus / resources.nnodes)
    return resources.nnodes, cores_per_node, gpus_per_node


@dataclass
class Partition:
    uri: str
    nnodes: int
    cores_per_node: int
    gpus_per_node: int

    @classmethod
    def from_uri(cls, uri):
        res = get_allocation_resources(uri)
        return cls(uri=uri, nnodes=res[0], cores_per_node=res[1], gpus_per_node=res[2])


class JobList(list):
    """
    @brief A list of 'AMSJobs'
    """

    def append(self, job: AMSJob):
        if not isinstance(job, AMSJob):
            raise TypeError("{self.__classs__.__name__} expects an item of a job")

        super().append(job)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if not isinstance(value, AMSJob):
            raise TypeError("{self.__classs__.__name__} expects an item of a job")

        super().__setitem__(index, value)


class AMSWorkflowManager:
    """
    @brief Manages all job submissions of the current execution.
    """

    def __init__(
        self,
        rmq_config: str,
        db_dir: str,
        application_name: str,
        url,
        domain_jobs: JobList,
        stage_jobs: JobList,
        sub_select_jobs: JobList,
        train_jobs: JobList,
    ):
        self._rmq_config = rmq_config
        self._db_dir = db_dir
        self._url = url
        self._application_name = application_name
        self._domain_jobs = domain_jobs
        self._stage_jobs = stage_jobs
        self._sub_select_jobs = sub_select_jobs
        self._train_jobs = train_jobs

    @property
    def rmq_config(self):
        return self._rmq_config

    def __str__(self):
        out = ""
        for job in self._domain_jobs:
            out += str(job) + "\n"

        for job in self._stage_jobs:
            out += str(job) + "\n"

        for job in self._sub_select_jobs:
            out += str(job) + "\n"

        for job in self._train_jobs:
            out += str(job) + "\n"

        return out

    def broadcast_train_specs(self, rmq_config):
        with AMSSyncProducer(
            rmq_config.service_host,
            rmq_config.service_port,
            rmq_config.rabbitmq_vhost,
            rmq_config.rabbitmq_user,
            rmq_config.rabbitmq_password,
            rmq_config.rabbitmq_cert,
            rmq_config.rabbitmq_ml_submit_queue,
        ) as rmq_fd:
            for ml_job in self._train_jobs:
                request = json.dumps(
                    [
                        {
                            "domain_name": ml_job.domain,
                            "job_type": "train",
                            "spec": ml_job.to_dict(),
                            "ams_log": True,
                            "request_type": "register_job_spec",
                        }
                    ]
                )
                rmq_fd.send_message(request)

            for subselect_job in self._sub_select_jobs:
                request = json.dumps(
                    [
                        {
                            "domain_name": subselect_job.domain,
                            "job_type": "sub_select",
                            "spec": subselect_job.to_dict(),
                            "ams_log": True,
                            "request_type": "register_job_spec",
                        }
                    ]
                )
                rmq_fd.send_message(request)

    def stager_done_cb(self, future):
        job_id = future.jobid()
        print(f"Stager got jobid {job_id}")

    def domain_jobid_cb(self, future):
        job_id = future.jobid()
        print(f"Got domain job id of {job_id} for AMSJob {future.ams_id}")
        self._domain_jobs[future.ams_id].flux_job_id = job_id

    def domain_done_cb(self, future):
        job_id = future.jobid()
        print(f"AMS Domain {future.ams_id} with JobID:{job_id} is done")

    def start_domain(self, store, rmq_config, domain_uri, rmq_stage_poller):
        def RMQServerStatusReport(stop_event):
            current_msg_count = 0
            poll_freq = 1
            while True:
                is_set = stop_event.is_set()
                prev_msg_count = current_msg_count
                current_msg_count = rmq_stage_poller.getMessageCount(
                    rmq_config.rabbitmq_queue_physics
                )
                print(
                    f"Server has {current_msg_count} messages in queue, {abs(current_msg_count - prev_msg_count)/poll_freq} msg/s"
                )
                if current_msg_count == 0 and is_set:
                    break
                time.sleep(poll_freq)
            print("Done waiting")

        # Create the stop event
        stop_event = threading.Event()
        # Start the background thread
        thread = threading.Thread(target=RMQServerStatusReport, args=(stop_event,))
        thread.start()

        print("Start Domain")
        self.jobs = {}
        with AMSFluxExecutor(
            False, threads=1, handle_args=(domain_uri,)
        ) as domain_executor:
            handle = flux.Flux(url=domain_uri)
            domain_futures = []
            for i, domain_job in enumerate(self._domain_jobs):
                # This is a hook, it will update the 'AMS_OBJECTS' env variable to point
                # to the latest model so the submitted job will pick it up.
                # It is not great, as it requires a piped execution.
                domain_job.ams_id = i
                domain_job.precede_deploy(store, rmq_config)
                domain_future = domain_executor.submit(domain_job.to_flux_jobspec())
                domain_future.ams_id = i
                domain_future = domain_future.add_jobid_callback(self.domain_jobid_cb)
                domain_future = domain_future.add_done_callback(self.domain_done_cb)
                domain_futures.append(domain_future)
            print("Moving to join")

            for i, domain_future in enumerate(domain_futures):
                try:
                    result = domain_future.result()
                    rpc_handle = flux.job.job_list_id(
                        handle, self._domain_jobs[i].flux_job_id
                    )
                    ji = rpc_handle.get_jobinfo()
                    results = ji.to_dict(False)
                    rt = results.get("runtime", -1)
                    success = results.get("success", False)
                    print(
                        f"AMS Domain {i} with JobID:{self._domain_jobs[i].flux_job_id} Success: {success} Duration: {rt}"
                    )
                except Exception as e:
                    print(e)
                    rpc_handle = flux.job.job_list_id(
                        handle, self._domain_jobs[i].flux_job_id
                    )
                    ji = rpc_handle.get_jobinfo()
                    print("Failed JOB Info:", json.dumps(ji.to_dict(False), indent=6))
            print("Going to shutdown")
            # Signal the thread to stop
            stop_event.set()
            thread.join()
            domain_executor.shutdown(wait=True)

    def start_stagers(
        self,
        store,
        rmq_config,
        domain_uri,
        stage_uri,
        orchestrator_publisher,
        rmq_stage_poller,
    ):
        with AMSFluxExecutor(
            False, threads=1, handle_args=(stage_uri,)
        ) as stager_executor:
            print("Connected to stager executor", stage_uri)
            # Spawn all stagers
            stager_futures = set()
            for stager in self._stage_jobs:
                print("Stager command is:", " ".join(stager.generate_cli_command()))
                stager_future = stager_executor.submit(stager.to_flux_jobspec())
                stager_futures.add(stager_future)
                job_id = stager_future.jobid()
                print(f"Stager JOB-ID is  {job_id}")
            print("Done scheduling stagers")
            self.start_domain(store, rmq_config, domain_uri, rmq_stage_poller)
            # AFAIK we need this cause here we lose a message.
            orchestrator_publisher.broadcast(json.dumps({"request_type": "terminate"}))
            stager_executor.shutdown(wait=True)

    def start(self, domain_uri, stage_uri, ml_uri):
        ams_orchestartor_job = AMSOrchestratorJob(ml_uri, self.rmq_config)
        rmq_config = AMSRMQConfiguration.from_json(self.rmq_config)
        print(f"Starting ..... {ml_uri} ... {stage_uri} ... {domain_uri}")
        with AMSDataStore(self._application_name, self._url) as store:
            print("Opened the AMS Store")
            # We start first the ML as we want to terminate only
            # after we have trained all the models.
            with AMSFluxExecutor(
                False, threads=1, handle_args=(ml_uri,)
            ) as ml_executor:
                print("Connected to ml executor")
                # The AMSFanOutProducer enables us to send control message to all stagers and
                # ml trainers. Currently
                with AMSFanOutProducer(
                    rmq_config.service_host,
                    rmq_config.service_port,
                    rmq_config.rabbitmq_vhost,
                    rmq_config.rabbitmq_user,
                    rmq_config.rabbitmq_password,
                    rmq_config.rabbitmq_cert,
                ) as orchestrator_publisher:
                    ml_future = ml_executor.submit(
                        ams_orchestartor_job.to_flux_jobspec()
                    )
                    job_id = ml_future.jobid()
                    print("ML JOB ID is:", job_id)
                    # We broadcast the training specification ...
                    self.broadcast_train_specs(rmq_config)
                    print("Broadcasted specs")
                    # We connect a status poller to the server. This will give us the load
                    # of the rmq_stager server.
                    with StatusPoller(
                        rmq_config.service_host,
                        rmq_config.service_port,
                        rmq_config.rabbitmq_vhost,
                        rmq_config.rabbitmq_user,
                        rmq_config.rabbitmq_password,
                        rmq_config.rabbitmq_cert,
                    ) as RMQPoll:
                        # Then we start the stagers. Stagers need to come online
                        # after the model server is up and running.
                        self.start_stagers(
                            store,
                            rmq_config,
                            domain_uri,
                            stage_uri,
                            orchestrator_publisher,
                            RMQPoll,
                        )
                ml_executor.shutdown(wait=True)

    @classmethod
    def from_descr(
        cls,
        json_file: str,
        rmq_config: Optional[str] = None,
    ):
        def collect_domains(jobs: JobList) -> Set[str]:
            return {job.domain for job in jobs}

        def create_domain_list(domains: List[Dict]) -> JobList:
            jobs = JobList()
            for job_descr in domains:
                jobs.append(AMSDomainJob.from_descr(job_descr))
            return jobs

        if not Path(json_file).exists():
            raise RuntimeError(f"Workflow description file {json_file} does not exist")

        with open(json_file, "r") as fd:
            data = json.load(fd)

        if "db" not in data:
            raise KeyError("Workflow decsription file misses 'db' description")

        if not all(key in data["db"] for key in {"url", "application-name", "dir"}):
            raise KeyError("Workflow description files misses entries in 'db'")

        db_dir = data["db"]["dir"]
        create_store_directories(db_dir)
        store = AMSDataStore(data["db"]["application-name"], data["db"]["url"]).open()

        if "domain-jobs" not in data:
            raise KeyError("Workflow description files misses 'domain-jobs' entry")

        if len(data["domain-jobs"]) == 0:
            raise RuntimeError(
                "There are no jobs described in workflow description file"
            )

        domain_jobs = create_domain_list(data["domain-jobs"])
        ams_rmq_config = AMSRMQConfiguration.from_json(rmq_config)
        for job in domain_jobs:
            job.db_dir = db_dir
            job.precede_deploy(store, ams_rmq_config)

        if "stage-job" not in data:
            raise RuntimeError("There is no description for a stage-job")

        stage_type = data["stage-job"].pop("type", "rmq")
        num_instances = data["stage-job"].pop("instances", 1)

        assert stage_type == "rmq", "We only support 'rmq' stagers"

        stage_resources = AMSJobResources(
            nodes=1, tasks_per_node=num_instances, cores_per_task=6, gpus_per_task=0
        )
        stage_jobs = JobList()
        stage_job = AMSNetworkStageJob.from_descr(
            data["stage-job"],
            str(Path(db_dir).resolve() / Path("candidates")),
            data["db"]["url"],
            data["db"]["application-name"],
            rmq_config,
            stage_resources,
        )
        # NOTE: We need to always copy in our environment. To make sure we find the respective packages
        stage_job.environ = os.environ
        stage_job.stdout = "stager_test.out"
        stage_job.stderr = "stager_test.err"
        print("Stager command is:", " ".join(stage_job.generate_cli_command()))
        stage_jobs.append(stage_job)

        sub_select_jobs = JobList()
        assert "sub-select-jobs" in data, "We are expecting a subselection job"
        for sjob in data["sub-select-jobs"]:
            sub_select_jobs.append(AMSSubSelectJob.from_descr(store, sjob))

        sub_select_domains = collect_domains(sub_select_jobs)

        assert "train-jobs" in data, "We are expecting training jobs"

        train_jobs = JobList()
        for sjob in data["train-jobs"]:
            train_jobs.append(AMSMLTrainJob.from_descr(store, sjob))

        train_domains = collect_domains(train_jobs)
        wf_domain_names = []
        for job in domain_jobs:
            wf_domain_names.append(*job.domain_names)

        wf_domain_names = list(set(wf_domain_names))

        for domain in wf_domain_names:
            print(domain)
            assert (
                domain in train_domains
            ), f"Domain {domain} misses a train description"
            assert (
                domain in sub_select_domains
            ), f"Domain {domain} misses a subselection description"
        store.close()
        store = AMSDataStore(data["db"]["application-name"], data["db"]["url"]).open()

        return cls(
            rmq_config,
            db_dir,
            data["db"]["application-name"],
            data["db"]["url"],
            domain_jobs,
            stage_jobs,
            sub_select_jobs,
            train_jobs,
        )
