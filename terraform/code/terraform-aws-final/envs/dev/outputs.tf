output "instance_public_ip" {
    value = module.compute.public_ip
}

output "instance_public_dns" {
    value = module.compute.public_dns  
}

output "bucket_name" {
    value = module.storage.bucket_name  
}