data "aws_subnets" "default_vpc_subnets" {
    filter {
      name = "vpc-id"
      values = [var.vpc_id]
    }
}

data "aws_ami" "amazon_linux" {

    most_recent = true
    owners = ["amazon"]
    filter {
      name = "name"
      values = ["al2023-ami-2023.*x86_64"]
    }

    filter {
      name = "virtualization-type"
      values = ["hvm"]
    }

    filter {
      name = "root-device-type"
      values = ["ebs"]
    }

}


resource "aws_instance" "web" {
    ami = data.aws_ami.amazon_linux.id
    instance_type = var.instance_type
    subnet_id = data.aws_subnets.default_vpc_subnets.ids[0]
    vpc_security_group_ids = [var.security_group_id]
    associate_public_ip_address = true
    tags = merge(var.tags, {
        Name = "${var.name_prefix}-ec2"
    })
  
}



