FROM python:latest

WORKDIR /tmp/install

RUN curl -o master.zip https://codeload.github.com/inspirai/wilderness-scavenger/zip/refs/heads/master
RUN unzip master.zip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -U pip
RUN pip install requests
RUN pip install wilderness-scavenger-master/.

# add your additional python denpendencies in the requirements.txt file
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# ========================================================
# Install other non-python dependencies here (if any)
# ========================================================

# DO NOT MODIFY BELOW THIS LINE
WORKDIR /home/inspirai
COPY submission submission
COPY eval.py eval.py
