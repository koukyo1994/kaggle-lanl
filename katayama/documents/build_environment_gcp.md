## Connect server via ssh
ssh katay@34.85.34.9 -i ~/.ssh/id_rsa

## gcloud cheet sheet
gcloud projects list
gcloud config set project [project id]
gcloud compute instances stop kaggle-instance

## jupyter notebook
docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle/python jupyter notebook --no-browser --ip="0.0.0.0" --notebook-dir=/tmp/working --allow-root

docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle-lanl jupyter notebook --no-browser --notebook-dir=/tmp/working --allow-root
