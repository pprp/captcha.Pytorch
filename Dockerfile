from conda/miniconda3-centos7
workdir /code
copy . /code
run pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
