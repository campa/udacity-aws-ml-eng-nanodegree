import json
import sagemaker
import base64
# from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT='image-classification-2024-04-26-12-39-22-832'## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    print("Event:", event.keys())
    image = base64.b64decode(event.get("body")['image_data'])

    # Instantiate a Predictor ## TODO: fill in
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT, sagemaker_session=sagemaker.Session())
 
    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = sagemaker.serializers.IdentitySerializer("image/png")
    
    # Make a prediction:
    # inferences = ## TODO: fill in
    # inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }