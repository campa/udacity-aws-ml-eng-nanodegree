#!/bin/sh

## To intercat with aws using cli
export AWS_ACCESS_KEY_ID=TBC
export AWS_SECRET_ACCESS_KEY=TBC
export AWS_SESSION_TOKEN=TBC
export AWS_DEFAULT_REGION=us-west-1
aws ec2 describe-instances --region us-west-1

# Package the Lambda with dependecies
echo Package the serializeImageData Lambda with dependecies   
cd lambdas/serializeImageData
rm -rf package
mkdir package
cp lambda_function.py package/
pip3 install --target ./package boto3
cd package
rm -rf ../*.zip
zip -r ../lambda_serializeImageData_package.zip .

# aws lambda create-function --function-name serializeImageData --runtime python3.12 --handler lambda_function.lambda_handler --role arn:aws:iam::372206764755:role/service-role/serializeImageData-role-0cbdis32 --zip-file fileb://../lambda_serializeImageData_package.zip

cd ../../..

aws lambda update-function-code --function-name serializeImageData \
--zip-file fileb://./lambdas/serializeImageData/lambda_serializeImageData_package.zip

## Used docker way due to be able to force linux/amd64, on my M2 apple notebook :( , ) platform for numpy indirect dependency
echo Package the classifyEncodedImg Lambda with dependecies   
cd lambdas/classifyEncodedImg-docker
docker build --platform linux/amd64 -t classify-encoded .  
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 372206764755.dkr.ecr.us-east-1.amazonaws.com
docker tag classify-encoded:latest 372206764755.dkr.ecr.us-east-1.amazonaws.com/udacity-classify-encoded:latest
docker push 372206764755.dkr.ecr.us-east-1.amazonaws.com/udacity-classify-encoded:latest
# aws lambda update-function-code \
#    --function-name  arn:aws:lambda:us-east-1:372206764755:function:classifyEncodedImg \
#    --region $AWS_DEFAULT_REGION \
#    --image-uri 372206764755.dkr.ecr.us-east-1.amazonaws.com/udacity-classify-encoded:latest

cd ../../..




