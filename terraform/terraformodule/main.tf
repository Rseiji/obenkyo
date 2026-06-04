provider "aws" {
  region = "us-east-1"
}

variable "buckets" {
  type = set(string)
  default = [ "seijibucket123-dev", "seijibucket123-test", "seijibucket123-prod" ]
}

module "my_bucket" {
  source = "./modules/s3"
  for_each = var.buckets
  bucket_name = each.value
  environment = each.key
  owner = "Seiji"
}