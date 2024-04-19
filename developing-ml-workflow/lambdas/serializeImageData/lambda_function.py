import json
import boto3
import base64
import io

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    #print(event)
    
    #print(s3.list_buckets())
    
    # Get the s3 address from the Step Function event input
    key = event.get('s3_key') ## TODO: fill in
    bucket =event.get('s3_bucket')  ## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    object = s3.get_object(Bucket=bucket, Key=key)
    
    # content = io.BytesIO(object['Body'].read())
    image_data = base64.b64encode(object['Body'].read())
    
    # file_name = "tmp/image.png"
    # s3.put_object(Body=content, Bucket=bucket, Key=file_name) 
    
    # We read the data from a file
    # with open(file_name, "rb") as f:
    #    image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }