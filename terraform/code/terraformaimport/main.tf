terraform {
  
  required_providers {
    aws = { source = "hashicorp/aws"}
  }
}

provider "aws" {
    region = "us-east-1"
 
}


import {
  to = aws_s3_bucket.bucket_importado
  id = "meu-bucket-terraform-7857"
}