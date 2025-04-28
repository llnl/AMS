# Container with all necessary AMS dependencies installed and with AMS 

The container runs a RMQ and a MariaDB server on startup. To build issue:

```bash
docker build -t <your-registry>/ams-tutorial:latest . --build-arg hypre_version=2.33.0
```

To run issue:
```bash
docker run --rm -it -v "$(pwd)":/workspace -w /workspace  <your-registry>/ams-tutorial:latest bash
```

