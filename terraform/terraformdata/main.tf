provider "aws" {
    region = "us-east-1"
}

variable "bucket_name" {
  description = "bucket's name"
  type = string
  default = "higaseijis_bucket_1234567890"
}

variable "Version" {
  description = "resource's version"
  type = number
}

variable "environment" {
  description = "environment"
  type = string
  default = "dev"
}

variable "owner" {
  description = "responsible for resource"
  type = string
  default = "seiji"
}

variable "kms_key_alias" {
  description = "alias for kms key"
  type = string
  default = "alias/aws/s3"
}

data "aws_kms_key" "existing_key" {
  key_id = var.kms_key_alias
}

locals {
  bucket_final_name = "${var.bucket_name}-${var.environment}"
  common_tags = {
    Name = "bucket demo terraform"
    Environment = var.environment
    Owner = var.owner
    Version = tostring(var.Version)
  }
}

resource "aws_s3_bucket" "my_test_bucket_2" {
  bucket = local.bucket_final_name
  tags = local.common_tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bucket_crypto" {
  bucket = aws_s3_bucket.my_test_bucket_2.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = data.aws_kms_key.existing_key.arn
      sse_algorithm = "aws:kms"
    }
  }
}