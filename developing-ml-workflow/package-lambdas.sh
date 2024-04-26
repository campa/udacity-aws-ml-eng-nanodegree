#!/bin/sh
# Package the Lambda with dependecies
echo Package the serializeImageData Lambda with dependecies   
cd lambdas/serializeImageData
rm -rf package
mkdir package
cp lambda_function.py package/
pip install --target ./package boto3
cd package
rm -rf ../*.zip
zip -r ../lambda_serializeImageData_package.zip .

cd ../../..

echo Package the classifyEncodedImg Lambda with dependecies   
cd lambdas/classifyEncodedImg
rm -rf package
mkdir package
cp lambda_function.py package/
pip install --target ./package sagemaker
cd package
## Reduce the size
find . -type d -name "tests" -exec rm -rfv {} +
find . -type d -name "__pycache__" -exec rm -rfv {} +
rm -rf ../*.zip
zip -r ../lambda_classifyEncodedImg_package.zip .

cd ../../..
