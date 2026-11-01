@echo off
setlocal

set DATASET_PATH=%~1
set BUCKET_NAME=%~2
set S3_PREFIX=%~3

if "%DATASET_PATH%"=="" set DATASET_PATH=data\rag_eval_dataset_v2.json
if "%S3_PREFIX%"=="" set S3_PREFIX=pra-prod/testing/rag-eval

if "%BUCKET_NAME%"=="" (
  echo Usage: scripts\upload_rag_eval_dataset.cmd [dataset_path] bucket_name [s3_prefix]
  echo Example: scripts\upload_rag_eval_dataset.cmd data\rag_eval_dataset_v2.json personalresearchassistantst-documentbucketae41e5a9-ht5svksrnpk5
  exit /b 1
)

set S3_KEY=%S3_PREFIX%/rag_eval_dataset_v2.json
set S3_URI=s3://%BUCKET_NAME%/%S3_KEY%

echo Uploading %DATASET_PATH% to %S3_URI%
aws s3 cp "%DATASET_PATH%" "%S3_URI%" --content-type application/json
if errorlevel 1 exit /b 1

echo Verifying %S3_URI%
aws s3api head-object --bucket "%BUCKET_NAME%" --key "%S3_KEY%"
