provider "aws" {
    region = var.aws_region  
}

locals {
  tags = merge(var.common_tags,{
  Project= "curso-terraform-pipelines"
  Environment ="dev"})
}

module "network" {
    source = "../../modules/network"
    name_prefix = "curso-terraform-dev"
    allowed_ssh_cidr = var.allowed_ssh_cidr
    http_port = var.http_port
    tags = local.tags
}

module "compute"{
    source = "../../modules/compute"
    name_prefix = "curso-terraform-dev"
    vpc_id = module.network.vpc_id
    security_group_id = module.network.security_group_id
    instance_type = var.instance_type
    http_port = var.http_port
    tags = local.tags
}

module "storage" {
    source = "../../modules/storage"
    bucket_name = var.bucket_name
    enable_versioning = var.enable_versioning
    tags = local.tags
}