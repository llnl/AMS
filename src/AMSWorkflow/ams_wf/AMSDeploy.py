import argparse
import os
from ams.ams_flux import AMSFluxExecutor
import time
from ams.ams_jobs import nested_instance_job_descr, get_echo_job
from ams.wf_manager import AMSWorkflowManager, get_allocation_resources

import warnings
from flux.job import FluxExecutor
import flux
from flux.job import kill as fkill


def verify_arg(name, uri, nodes):
    if uri is None and nodes is None:
        raise argparse.ArgumentError(None, f"{name} needs to either specify num nodes or an existing flux-uri")
    if uri is not None and nodes is not None:
        warnings.warn("We ignore the number of nodes in partion {name} as an existing uri is given")
        nodes = 0

    return uri, nodes


def get_partition_uri(root_executor, nnodes, cores_per_node, gpus_per_node, time):
    sleep_job = nested_instance_job_descr(nnodes, cores_per_node, gpus_per_node, time=time)
    fut = root_executor.submit(sleep_job)
    # NOTE: The following code is required to make sure the instance has fully started.
    # Got the code from @grondo
    uri = fut.uri()
    nested_instance = flux.Flux(uri)
    nested_instance.rpc("state-machine.wait").get()
    return uri, fut


def main():
    parser = argparse.ArgumentParser(description="AMS workflow deployment")
    parser.add_argument("--root-uri", help="Flux uri of an already existing allocation", required=True)
    parser.add_argument(
        "--ml-uri",
        help="Flux uri of an already existing allocation to schedule ml jobs on",
        required=False,
    )
    parser.add_argument(
        "--ml-nodes",
        help="Number of nodes to run ml jobs (required only when ml-uri is not set)",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--stage-uri",
        help="Flux uri of an already existing allocation to schedule stage jobs on",
        required=False,
    )
    parser.add_argument(
        "--stage-nodes",
        help="Number of node existing in the stage allocation",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--sleep-time",
        help="Time of nested allocations (used) for debugging",
        required=False,
        default="0",
    )

    parser.add_argument(
        "--workflow-descr",
        "-w",
        help="JSON file describing the workflow",
        required=True,
    )
    parser.add_argument("--credentials", "-c", help="JSON file describing the workflow", required=True)

    args = parser.parse_args()

    sleep_time = int(args.sleep_time)
    if sleep_time == 0:
        start = int(os.getenv("SLURM_JOB_START_TIME", "0"))
        end = int(os.getenv("SLURM_JOB_END_TIME", "0"))
        sleep_time = end - start

    if sleep_time == 0:
        print("Cannot create background job with 0 time")
        return
    print(f"Partions will be allocated for {sleep_time}")

    wf_manager = AMSWorkflowManager.from_descr(args.workflow_descr, args.credentials)
    print(wf_manager)

    stage_uri, num_stage_nodes = verify_arg("stage", args.stage_uri, args.stage_nodes)
    ml_uri, num_ml_nodes = verify_arg("ml", args.ml_uri, args.ml_nodes)

    # I need to know the number of cores so that we can split the partions accordingly.
    nnodes, cores_per_node, gpus_per_node = get_allocation_resources(args.root_uri)

    print(
        "Root Allocation has NNodes:{0} NCores (per Node):{1} NGPUs (per Node):{2}".format(
            nnodes, cores_per_node, gpus_per_node
        )
    )

    assert nnodes > num_stage_nodes + num_ml_nodes, "Insufficient resources to schedule workflow"

    num_domain_nodes = nnodes - num_ml_nodes - num_stage_nodes

    print(f"ML Allocation has NNodes:{num_ml_nodes}")
    print(f"Stage Allocation has NNodes:{num_stage_nodes}")
    print(f"Domain Allocation has NNodes:{num_domain_nodes}")

    # NOTE: We need a AMSFluxExecutor to easily get flux uri because FluxExecutor does not provide the respective API
    # We set track_uri to true to enable the executor to generate futures tracking the uri of submitted jobs
    start = time.time()
    with AMSFluxExecutor(True, threads=1, handle_args=(args.root_uri,)) as root_executor:
        print("Spawning Flux executor for root took", time.time() - start)
        start = time.time()
        domain_uri, domain_future = get_partition_uri(
            root_executor,
            num_domain_nodes,
            cores_per_node,
            gpus_per_node,
            str(sleep_time),
        )
        print(f"Resolving domain uri took {domain_uri}", time.time() - start)
        domain_res = get_allocation_resources(domain_uri)
        print("Domain resources are ", domain_res)
        start = time.time()
        ml_future = None
        stage_future = None

        if ml_uri is None:
            (
                ml_uri,
                ml_future,
            ) = get_partition_uri(
                root_executor,
                num_ml_nodes,
                cores_per_node,
                gpus_per_node,
                str(sleep_time),
            )
        print("Resolving ML uri  took", time.time() - start, ml_uri)
        start = time.time()

        if stage_uri is None:
            stage_uri, stage_future = get_partition_uri(
                root_executor,
                num_stage_nodes,
                cores_per_node,
                gpus_per_node,
                str(sleep_time),
            )
        print("Resolving stage uri  took", time.time() - start, stage_uri)

        # 1) We first schedule the ML training orchestrator.
        wf_manager.start(domain_uri, stage_uri, ml_uri)
        # The root executor should not wait, the partitions have "infinite allocation time. So we forcefully shut them down"
        print("All internal executors are done ... moving to stopping root job")
        # TODO: When I get here I need to kill all the jobs of the partitions and exit.
        print("Stopping domain partition...")
        domain_handle = flux.Flux(domain_uri)
        domain_handle.rpc("shutdown.start").get()
        # fkill(domain_handle, domain_future.jobid())

        if ml_uri:
            print("Stopping ml partition")
            ml_handle = flux.Flux(ml_uri)
            ml_handle.rpc("shutdown.start").get()
        if stage_uri:
            print("Stopping stager partition")
            stage_handle = flux.Flux(stage_uri)
            stage_handle.rpc("shutdown.start").get()
        print("Shutting root executor")
        root_executor.shutdown(wait=False, cancel_futures=True)
        print("All daemons are down")
    print("Exiting")

    return 0


if __name__ == "__main__":
    sys.exit(main())
