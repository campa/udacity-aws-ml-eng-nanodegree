import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-01-31-12-26-05-210"## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    #image = base64.b64decode(## TODO: fill in)
    print(event)
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor ## TODO: fill in
    predictor = sagemaker.predictor.Predictor( ENDPOINT, sagemaker_session=sagemaker.Session())
 
    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    # inferences = ## TODO: fill in
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }