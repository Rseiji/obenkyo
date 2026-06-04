provider "aws" {
    region = "us-east-1"  
}

variable "bucket_name" {
    description = "Nome base para o bucket"
    type = string
    default = "bucket-teste-data-77889"  
}

variable "versao" {
    description = "Versão do recurso"
    type = number  
}

variable "environment" {
    description = "Ambinte"
    type = string
    default = "dev"  
}

variable "owner" {
    description = "Responsável pelo recurso"
    type = string
    default = "Fernando"
}

variable "kms_key_alias" {
    description = "Alias de uma chave KMS existente"
    type = string
    default = "alias/aws/s3"  
}

data "aws_kms_key" "chave_existente"{
    key_id = var.kms_key_alias
}

locals {
  bucket_final_name = "${var.bucket_name}-${var.environment}"
  common_tags = {
    Name = "Bucket demo terraform"
    Environment = var.environment
    Owner = var.owner
    Version = tostring(var.versao)
  }
}

resource "aws_s3_bucket" "meu_bucket_test" {
    bucket = local.bucket_final_name
    tags = local.common_tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bucket_crypto" {
    bucket = aws_s3_bucket.meu_bucket_test.id
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = data.aws_kms_key.chave_existente.arn
        sse_algorithm = "aws:kms"
      }
    }
  
}