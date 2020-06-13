FROM continuumio/miniconda3:4.7.10

MAINTAINER Danny Goldstein <dgold@caltech.edu>

# Install astromatic software
RUN conda install -c conda-forge astromatic-swarp astromatic-source-extractor

# Install hotpants
RUN apt-get update && apt-get install -y libcfitsio-dev libcurl4-openssl-dev postgresql-client postgresql \
    libpq-dev make gcc libbz2-dev curl && \
    git clone https://github.com/zuds-survey/hotpants.git && \
    cd hotpants && LIBS="-lm -lcfitsio -lcurl" make -e && \
    cp hotpants $CONDA_PREFIX/bin && cd -


SHELL ["/bin/bash", "-c"]

RUN mkdir zuds-pipeline && mkdir zuds-pipeline/zuds
ADD setup.py requirements.txt zuds-pipeline/
ADD zuds zuds-pipeline/zuds/

RUN pip --no-cache-dir install jupyter && cd zuds-pipeline && \
    pip install --no-cache-dir . && cd -

RUN curl https://portal.nersc.gov/cfs/m937/demo.tar.gz -o demo.tar.gz && \
    mkdir ~/.data && cd ~/.data && tar -xvzf ../demo.tar.gz && \
    rm ../demo.tar.gz

ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
