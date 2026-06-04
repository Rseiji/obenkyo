variable "bucket_name" {
    type = string
}

variable "enable_versioning" {
    type = bool  
}

variable "tags" {
    type = map(string)
  
}