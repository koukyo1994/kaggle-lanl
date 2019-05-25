### Dockerをインストール
sudo amazon-linux-extras install docker
sudo usermod -g docker ec2-user
sudo /bin/systemctl restart docker.service

### Gitをインストール
sudo yum install git -y

### pyenvをインストール
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile

### Python3系のインストール
sudo yum install gcc zlib-devel bzip2 bzip2-devel readline readline-devel sqlite sqlite-devel openssl openssl-devel -y
pyenv install 3.6.8
pyenv global 3.6.8
pyenv rehash

### pipのインストール
python get-pip.py --user

### Install kaggle api
pip install kaggle --user
mkdir ~/.kaggle
vim ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

### Pull docker image
docker login
docker pull fozzhey/kaggle-lanl


### 便利コマンド
sudo service docker start

docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it fozzhey/kaggle-lanl jupyter notebook --no-browser --notebook-dir=/tmp/working --allow-root
