provider "aws" {
    region = "us-east-1"
}

variable "buckets" {
    type = set(string)
    default = [ "bucket-modules-764-dev","bucket-modules-764-test","bucket-modules-764-prod" ]
  
}

module "meu_bucket" {
    source = "./modules/s3"
    for_each = var.buckets
    bucket_name = each.value
    environment = each.key
    owner = "Fernando"
  
}