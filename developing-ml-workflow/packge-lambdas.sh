#!/bin/sh
# Package the Lambda with dependecies
echo Package the Lambdas with dependecies   
cd lambdas/serializeImageData
mkdir package
pip install --target ./package boto3
cd package
zip -r ../lambda_serializeImageData_package.zip .
