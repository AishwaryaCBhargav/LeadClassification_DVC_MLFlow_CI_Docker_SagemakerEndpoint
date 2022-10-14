import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import json
import boto3

global app_name
global region
app_name = 'mlpipeline'
region = 'us-east-1'

def check_status(app_name):
    sage_client = boto3.client('sagemaker', region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status

def query_endpoint(app_name, input_json):
    client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType='application/json; format=pandas-split',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds

## check endpoint status
print("Application status is: {}".format(check_status(app_name)))

# Prepare data to give for predictions

test_data = pd.read_csv('data/sample_preprocessed_data.csv', sep=",")

## create test data and make inference from enpoint
query_input = test_data.iloc[[3]].to_json(orient="split")
print(query_input)
prediction = query_endpoint(app_name=app_name, input_json=query_input)  