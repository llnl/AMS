FROM fluxrm/flux-core:latest AS base
USER root
SHELL ["/bin/bash", "-lc"]

ARG num_jobs
ARG hypre_version

# Install MariaDB + tools
# Install extra packages (e.g., git, python3-pip)
RUN apt-get update && \
    apt-get install -y lsb-release gnupg software-properties-common  \
    			git python3 python3-pip mariadb-server mariadb-client \
			curl \
			rabbitmq-server supervisor \
			python3-venv \
			nlohmann-json3-dev libjson-glib-dev \
			build-essential cmake \
		        libopenmpi-dev libmetis-dev libhypre-dev \
		        libblas-dev liblapack-dev \
			libhdf5-dev hdf5-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY rabbitmq.conf /etc/rabbitmq/rabbitmq.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /run/mysqld \
 && chown mysql:mysql /run/mysqld

RUN rabbitmq-plugins enable --offline rabbitmq_management


ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]






























































