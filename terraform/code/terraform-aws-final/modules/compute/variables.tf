variable "name_prefix" {
    type = string  
}

variable "vpc_id" {
    type = string  
}

variable "security_group_id" {
    type = string  
}

variable "instance_type" {
    type = string  
}

variable "http_port" {
    type = number  
}

variable "tags" {
    type = map(string)  
}