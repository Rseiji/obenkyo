provider "aws" {
    region = "us-east-1"
}

variable "bucket_name" {
    description = "s3 bucket name"
    type = string
    default = "my-bucket-higa"
}

variable "Version" {
    description = "resource version"
    type = number
}

resource "aws_s3_bucket" "my_test_bucket" {
    bucket = var.bucket_name
    tags = {
        "name" = "demo bucket terraform"
        "env" = "dev"
        "Owner" = "Higa"
        "Version" = var.Version
    }
}

output "nome_bucket" {
    value = aws_s3_bucket.my_test_bucket.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.my_test_bucket.arn
}
