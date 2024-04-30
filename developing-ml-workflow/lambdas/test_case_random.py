import random
import boto3
import json

BUCKET = "sagemaker-us-east-1-372206764755"

def lambda_handler(event, context):
    # Setup s3 in boto3
    s3 = boto3.resource('s3')
    
    # Randomly pick from sfn or test folders in our bucket
    objects = s3.Bucket(BUCKET).objects.filter(Prefix="test/")
    
    # Grab any random object key from that folder!
    obj = random.choice([x.key for x in objects])
    
    return json.dumps({
        "image_data": "",
        "s3_bucket": BUCKET,
        "s3_key": obj
    })