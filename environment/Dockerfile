FROM sgvrcluster.kaist.ac.kr/woobin/conda:cuda11.6


ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="$PATH:/root/miniconda/bin/"

RUN conda create -n NSFF python=3.7.13
SHELL ["conda", "run", "-n", "NSFF", "/bin/bash", "-c"]
RUN echo "In a conda environment: $CONDA_DEFAULT_ENV"

COPY environment/torch-1.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl .
SHELL ["conda", "run", "-n", "NSFF", "/bin/bash", "-c"]
RUN pip install torch-1.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl
RUN rm torch-1.12.1%2Bcu116-cp37-cp37m-linux_x86_64.whl

SHELL ["conda", "run", "-n", "NSFF", "/bin/bash", "-c"]
RUN pip install torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

SHELL ["conda", "run", "-n", "NSFF", "/bin/bash", "-c"]
RUN pip install torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*

COPY environment/requirements.txt .
SHELL ["conda", "run", "-n", "NSFF", "/bin/bash", "-c"]
RUN pip install -r requirements.txt

RUN rm requirements.txt

RUN apt-get update && apt-get install git -y && rm -rf /var/lib/apt/lists/*

# for entry
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install PyYAML==6.0.1