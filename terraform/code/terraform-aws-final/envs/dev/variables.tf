variable "aws_region" {
    type = string  
}

variable "instance_type" {
    type = string  
}

variable "bucket_name" {
    type = string  
}

variable "allowed_ssh_cidr" {
    type = string  
}

variable "http_port" {
    type = number  
}

variable "enable_versioning" {
    type = bool  
}

variable "common_tags" {
    type = map(string)
  
}