FROM jupyter/pyspark-notebook:latest

USER root

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.0+cpu" && \
    \
    pip --default-timeout=600 install --no-cache-dir \
        -r /tmp/requirements.txt && \
    \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/requirements.txt

RUN mkdir -p /opt/exp /opt/data && \
    chown -R $NB_UID:$NB_GID /opt

USER $NB_UID

WORKDIR /opt/exp
ENV PYTHONPATH=/opt:/opt/exp

EXPOSE 8888
CMD ["start-notebook.sh"]