## Build compute Engine
- OSはUbuntuを選択
- 公開鍵を設定。 `cat ~/.ssh/id_rsa.pub`

## Connect server via ssh
ssh katay@35.213.24.98 -i ~/.ssh/id_rsa

## Install docker
sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get install -y docker-ce
sudo usermod -aG docker katay

## Install unzip
sudo apt-get install zip unzip

## Install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user

## Install kaggle api
pip install kaggle --user
mkdir ~/.kaggle
vim ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

## Pull docker image
docker login
docker pull fozzhey/kaggle-lanl

## Git clone
git clone https://github.com/koukyo1994/kaggle-lanl.git

## Download dataset
kaggle competitions download -c LANL-Earthquake-Predicttion
unzip test.zip
unzip train.csv.zip

## jupyter notebook
docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it fozzhey/kaggle-lanl jupyter notebook --no-browser --notebook-dir=/tmp/working --allow-root

## gcloud cheet sheet
gcloud projects list
gcloud config set project [project id]
gcloud compute instances stop kaggle-instance


scp -i ~/.ssh/id_rsa katay@35.213.24.98:/home/katay/kaggle-lanl/katayama/src/data/output/best_kernel/submission_50000_top100.csv ./git/private/kaggle-lanl/katayama/src/data/output/best_kernel/submission_50000_top100.csv
