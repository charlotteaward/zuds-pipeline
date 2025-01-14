FROM continuumio/miniconda3:4.7.10

MAINTAINER Danny Goldstein <dgold@caltech.edu>

# Install astromatic software
RUN conda install -c conda-forge astromatic-swarp astromatic-source-extractor \
    astromatic-scamp

RUN conda install pip 

# Install hotpants
RUN apt-get update && apt-get install -y libcfitsio-dev libcurl4-openssl-dev postgresql-client postgresql \
    libpq-dev make gcc libbz2-dev curl gfortran g++ && \
    git clone https://github.com/zuds-survey/hotpants.git && \
    cd hotpants && LIBS="-lm -lcfitsio -lcurl" make -e && \
    cp hotpants $CONDA_PREFIX/bin && cd -

# Install mpi4py so things can run at nersc
RUN curl -SL http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    -o mpich-3.2.tar.gz \
    && tar -xzf mpich-3.2.tar.gz \
    && cd mpich-3.2 \
    && CC="gcc" CXX="g++" CFLAGS="-O3 -fPIC -pthread" CXXFLAGS="-O3 -fPIC -pthread" ./configure  --prefix=/usr \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf mpich-3.2*


RUN curl -SL https://files.pythonhosted.org/packages/04/f5/a615603ce4ab7f40b65dba63759455e3da610d9a155d4d4cece1d8fd6706/mpi4py-3.0.2.tar.gz \
    -o mpi4py-3.0.2.tar.gz \
    && tar xzf mpi4py-3.0.2.tar.gz \
    && cd mpi4py-3.0.2 \
    && python setup.py build --mpicc="mpicc" --mpicxx="mpicxx" \
    && python setup.py install \
    && cd .. \
    && rm -rf mpi4py*


SHELL ["/bin/bash", "-c"]

RUN mkdir zuds-pipeline && mkdir zuds-pipeline/zuds
ADD setup.py requirements.txt zuds-pipeline/
ADD zuds zuds-pipeline/zuds/

RUN cd zuds-pipeline && pip install --no-cache-dir . && cd -
