FROM nagadomi/torch7:cuda10.1-cudnn7-devel-ubuntu18.04
RUN rm /etc/apt/sources.list.d/*

RUN apt-get update

RUN echo 'root:root' | chpasswd
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync vim python3-pip
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/\#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config

RUN git clone https://github.com/HiroIshida/inpaint.git
RUN cd inpaint && pip3 install -e . && git submodule update --init
