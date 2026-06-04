aws_region = "us-east-1"

instance_type = "t3.small"

bucket_name = "bucket-prod-pipeline-12345"

allowed_ssh_cidr = "0.0.0.0/0"

http_port = 80

enable_versioning = false

common_tags = {
  "Owner" = "Fernando"
  "Course" = "Terraform"
  "Managed" = "Terraform"
}