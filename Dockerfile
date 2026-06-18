FROM condaforge/miniforge3:26.1.1-3

WORKDIR /app/historoi

COPY environment.yml .

RUN mamba env create -f environment.yml && \
    mamba clean -afy

COPY . .

ENV PATH=/opt/conda/envs/historoi/bin:$PATH
ENV PYTHONUNBUFFERED=1

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN echo "conda activate historoi" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "historoi"]
CMD ["/bin/bash"]
