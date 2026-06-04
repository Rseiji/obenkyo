variable "bucket_name" {
    type = string  
}

variable "environment" {
    type = string
}

variable "owner" {
    type = string  
}

resource "aws_s3_bucket" "bucket" {
    bucket = var.bucket_name
    tags = {
      Name = var.bucket_name
      Environment = var.environment
      Owner = var.owner
      ManagedBy = "Terraform"
    } 
}

output "bucket_name" {
    value = aws_s3_bucket.bucket.bucket
}