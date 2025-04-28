# AMS Tutorial

The tutorial provides a docker with all the necessary installations to build applications against AMS and run them on a CPU system.

## Download container

To download the container please issue the following command on your terminal:

```
docker pull ghcr.io/llnl/ams-x86-tutorial:latest
```

The expected output should be similar to:

```
154ef9065217: Already exists
4e338da1ac28: Already exists
068a1074ab96: Already exists
dd6c369f83fa: Already exists
a112725ae27e: Already exists
c948d35905b3: Already exists
edcd2ec99d0b: Already exists

Digest: sha256:24950b5ebb5ee90657fdd17d007921732b36e7d8e3820a6778779ed9357d2b9f
Status: Downloaded newer image for ghcr.io/llnl/ams-x86-tutorial:latest
```

## Run container interactively

docker run --rm -it \
  -v "$(pwd)":/workspace -w /workspace \
  ghcr.io/llnl/ams-x86-tutorial:latest \
  bash

The command should provide an interactive `bash` shell and the output should look like the following:

```
250428 15:31:30 mysqld_safe Logging to syslog.
250428 15:31:30 mysqld_safe Starting mariadbd daemon with databases from /var/lib/mysql
mysqld is alive
MariaDB is up!
Root password set to 'root'
```

