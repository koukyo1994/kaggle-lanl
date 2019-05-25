import sys
import os
import io
import csv
import tempfile
import codecs
import configparser

import pandas as pd
import boto3
from boto3.session import Session
import s3fs

# os.chdir('src')

from paths import *


# Define the config for s3 select
s3select_params = {
    'ExpressionType':'SQL',
    'InputSerialization':{'CompressionType':'NONE', 'CSV':{'FileHeaderInfo':'Use', 'RecordDelimiter' : '\n', 'FieldDelimiter' : ','}},
    'OutputSerialization':{'CSV':{'RecordDelimiter':'\n', 'FieldDelimiter':','}}
}

# Create session
# session = Session(profile_name=profile_name)
session = Session(region_name='ap-northeast-1')
s3_resource = session.resource('s3')
s3_client = session.client('s3')

# Create s3filesystem instance
# fs = s3fs.S3FileSystem(anon=False, key=ACCESS_KEY_ID, secret=SECRET_ACCESS_KEY)
fs = s3fs.S3FileSystem(anon=False)


def download_data(bucket_name, key_path, download_path):
    # s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(key_path, download_path)

def upload_data(bucket_name, key_path, file_path):
    # s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(file_path, key_path)

def cp(from_file_path, to_file_path):
    from_bucket_name, from_key_path = extract_bucket_and_key_path(from_file_path)
    to_bucket_name, to_key_path = extract_bucket_and_key_path(to_file_path)

    # s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY, region_name='ap-northeast-1')
    s3_client.copy_object(Bucket=to_bucket_name, Key=to_key_path, CopySource={'Bucket': from_bucket_name, 'Key': from_key_path})

def delete(file_path):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    # s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY, region_name='ap-northeast-1')
    s3_client.delete_object(Bucket=bucket_name, Key=key_path)

def mv(from_file_path, to_file_path):
    cp(from_file_path, to_file_path)
    delete(from_file_path)

def ls(file_path, full_path=False):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    # client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    # s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key_path)

    filename_list = []
    for res in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key_path)['Contents']:
        filename_list.append(res['Key'].split('/')[-1])

    return filename_list

def exists(file_path):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    filename = key_path.split("/")[-1]
    dir_path = f's3://{bucket_name}/{"/".join(key_path.split("/")[:-1])}'

    for name in ls(dir_path)[1:]:
        if name == filename:
            return True
    return False

def select(file_path, query, header=None, FileHeaderInfo='USE', decode='utf-8'):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    params = s3select_params.copy()
    params['InputSerialization']['CSV']['FileHeaderInfo'] = FileHeaderInfo

    # s3_client = boto3.client('s3', 'ap-northeast-1', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    response = s3_client.select_object_content(Bucket=bucket_name, Key=key_path, Expression=query, **params)

    text = ''
    for event in response['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode(decode)
            text += records

    df = pd.read_csv(io.StringIO(text), header=None)
    if header is not None:
        df.columns = header

    return df

def read_excel_in_s3(file_path, **kwargs):
    #file_path, kwargs = "s3://clusteringai/output/cluster_stats/cluster_stats_20190319.xlsx", dict()
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_path = os.path.join(tmp_dir, key_path.split('/')[-1])
        download_data(bucket_name, key_path, download_path)
        df = pd.read_excel(download_path, **kwargs)

    return df

def read_csv_in_s3(file_path, **kwargs):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_path = os.path.join(tmp_dir, key_path.split('/')[-1])
        download_data(bucket_name, key_path, download_path)
        df = pd.read_csv(download_path, **kwargs)

    return df

def to_csv_in_s3(file_path, df, **kwargs):
    bucket_name, key_path = extract_bucket_and_key_path(file_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, key_path.split('/')[-1])
        df.to_csv(save_path, **kwargs)
        upload_data(bucket_name, key_path, save_path)

    return

def extract_bucket_and_key_path(file_path):
    bucket_name = file_path.split('//')[-1].split('/')[0]
    key_path = '/'.join(file_path.split('//')[-1].split('/')[1:])
    return bucket_name, key_path

def append_lines_to_csv_in_s3(key_path,append_line_list:list):
    """
    Write to a csv in append mode.
    append_line_list: list of list which contains the content to be appended.
    """
    with fs.open(key_path,mode="a") as f:
        writer = csv.writer(f,lineterminator="\n")
        for line in append_line_list:
            writer.writerow(line)
