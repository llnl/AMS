---
name: flux
description: >
  Commands for running and monitoring jobs on a Flux (flux-framework) HPC
  scheduler: submitting jobs, checking whether a job is running, reading job
  output and exit codes, and cancelling jobs. Use this whenever a task involves
  submitting, monitoring, inspecting, or cancelling work on a Flux cluster, or
  running builds / test suites / model serving on compute nodes rather than the
  login node. Trigger it on any mention of `flux submit`, `flux run`,
  `flux jobs`, `flux batch`, Flux job IDs, or "is my job running / done" on an
  HPC system where Flux is the main scheduler.
---

# Flux job management

Flux is a resource manager / scheduler used on HPC clusters. Use it to run
anything heavy — builds, test suites, model serving, data processing — as a job
on **compute nodes**. Do **not** run heavy work directly on the login node.

Queue names, bank/account, and node counts are cluster-specific. This skill
covers the generic commands only; get the per-machine values (queue, bank,
typical `-N`/`-n`) from the project's machine-specific setup notes before
submitting. In general, you want to use `--exclusive` when you request nodes.

## Interactive allocations

```bash
flux alloc -B <bank> --exclusive -N1 -q pdebug -t 1h # open a new shell with the allocated nodes
flux alloc -B <bank> --exclusive -N4 -t 8h
```

## Submitting jobs

```bash
flux submit ./script.sh          # queue a job; prints a job ID and returns immediately
flux run ./script.sh             # run interactively and block until it finishes
flux batch ./batch.sh            # submit a batch script (directives via '# flux:' lines)
flux submit -N2 -n8 ./script.sh  # request 2 nodes, 8 tasks
flux submit --queue=<queue> --name=<name> ./script.sh   # target a queue, name the job
```

A batch script declares its resources with `# flux:` directive lines, e.g.:

```bash
#!/bin/sh
# flux: -N4 -n16
flux run -n16 ./my_step.sh
```

## Checking whether a job is running

```bash
flux jobs                        # your active jobs (pending + running)
flux jobs -a                     # include completed / inactive jobs
flux jobs --filter=running       # only running jobs (also: pending, inactive)
flux job last                    # job ID of your most recent submission
```

## Reading output and exit code

```bash
flux submit --watch ./script.sh                  # stream output live as it runs
flux job attach $(flux job last)                 # attach to / print output of the last job
flux submit --output=job-{{id}}.out ./script.sh  # write stdout to a file named per job ID
flux jobs --no-header -o '{status}:{returncode}' <jobid>   # status + exit code of one job
```

A non-zero `returncode` means the job failed — inspect its output before assuming
the step succeeded. Do not report a job as "passed" until you have confirmed both
that it is `inactive` and that its return code is `0`.

## Cancelling jobs

```bash
flux cancel <jobid>              # cancel a single job
flux cancel --all                # cancel all of your jobs
flux cancel --states=RUN         # cancel jobs in a given state
```

## Inspecting resources

```bash
flux resource list               # nodes available to you and their state
flux uptime                      # is the Flux instance up, and for how long
flux overlay status              # health of the Flux overlay network
```

## Notes for agents

- Prefer `flux submit` (non-blocking) for long work, then poll with `flux jobs`;
  use `flux run` only for quick interactive checks.
- Always capture the job ID from `flux submit` (or use `flux job last`) so you can
  check status and output later.
- Never launch a full model-serving stack or a long test run on the login node —
  submit it as a Flux job.

Full command reference: https://flux-framework.org/cheat-sheet/
