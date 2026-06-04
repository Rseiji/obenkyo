variable "name_prefix" {
    type = string
}

variable "allowed_ssh_cidr" {
    type = string  
}

variable "http_port" {
    type = number  
}

variable "tags" {
    type = map(string)  
}