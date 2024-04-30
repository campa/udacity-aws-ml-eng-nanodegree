import json
import sagemaker
import base64
# from sagemaker.serializers import IdentitySerializer

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
    
    # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
   
    return {
        'statusCode': 200,
        'body': {
            "inferences": json.dumps(inferencesList)
        }
    }

