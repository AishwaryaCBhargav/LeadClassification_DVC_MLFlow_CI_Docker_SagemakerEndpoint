import mlflow.sagemaker as mfs

experiment_id = '1'
run_id = '999868bbf228434db942a9e87291a4dd'
region = 'us-east-1'
aws_id = '000548911229'
arn = 'arn:aws:iam::000548911229:role/aishwarya-sagemaker'
app_name = 'mlpipeline'
model_uri = 'mlruns/%s/%s/artifacts/xgbclassifier' % (experiment_id, run_id)
tag_id = 'latest'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pipeline:' + tag_id

mfs.deploy(app_name=app_name, 
           model_uri=model_uri, 
           region_name=region, 
           mode="create",
           execution_role_arn=arn,
           image_url=image_url)

# arn:aws:iam::000548911229:user/aishwarya