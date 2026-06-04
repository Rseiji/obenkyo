provider "aws" {
  region = "us-east-1"
}



resource "aws_s3_bucket" "meu_bucket_teste" {
    bucket = var.bucket_name
    tags = {
      "Name" = "Bucket demo terraform"
      "Environment" = "Dev"
      "Owner" = "Fernando Amaral"
      "Version" = var.versao
    }
  
}

output "nome_do_bucket" {
  value = aws_s3_bucket.meu_bucket_teste.bucket
}

output "bucket_arn" {
    value = aws_s3_bucket.meu_bucket_teste.arn
}