# __generated__ by Terraform
# Please review these resources and move them into your main configuration files.

# __generated__ by Terraform from "meu-bucket-terraform-7857"
resource "aws_s3_bucket" "bucket_importado" {
  bucket              = "meu-bucket-terraform-7857"
  bucket_namespace    = "global"
  force_destroy       = false
  object_lock_enabled = false
  region              = "us-east-1"
  tags = {
    Environment = "Dev"
    Name        = "Bucket demo terraform"
    Owner       = "Fernando Amaral"
    Version     = "3"
  }
}
