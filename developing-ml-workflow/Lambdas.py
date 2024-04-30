

### serialize lambda function
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

### ClassifyImg

import json
import sagemaker
import base64

# Fill this in with the name of your deployed model
ENDPOINT='cifar-images-endpoint'## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    print("Event:", event.keys())
    image = base64.b64decode(event.get("image_data"))

    # Instantiate a Predictor ## TODO: fill in
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT, sagemaker_session=sagemaker.Session())
 
    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = sagemaker.serializers.IdentitySerializer("image/png")
    
    # Make a prediction:
    # inferences = ## TODO: fill in
    inferences = predictor.predict(image)

    decoded_inferences_string = inferences.decode('utf-8')
    inferencesList = eval(decoded_inferences_string)

    print("inferencesList", type(inferencesList))
    print("inferencesList", inferencesList)
   
    return {
        'statusCode': 200,
        'body': {
            "inferences": json.dumps(inferences)
        }
    }

### filter confidence level lambda function
import json

THRESHOLD = .70

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferencesStr = event.get('inferences') ## TODO: fill in
    
    inferences = eval(inferencesStr)
    print("inferences", inferences)
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = all(inference > THRESHOLD for inference in inferences)  ## TODO: fill in
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }