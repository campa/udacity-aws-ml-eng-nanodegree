import json

THRESHOLD = .70

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event.get('inferences') ## TODO: fill in
    
    print("inferences", inferences)
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = all(int(inference) > THRESHOLD for inference in inferences)  ## TODO: fill in
    
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

