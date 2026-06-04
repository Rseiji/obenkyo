data "aws_vpc" "default" {
    default = true
}


resource "aws_security_group" "web" {
    name = "${var.name_prefix}-web-sg"
    description = "Security group da VM de Orquestracao"
    vpc_id = data.aws_vpc.default.id
    tags = merge(var.tags, {
        Name = "${var.name_prefix}-web-sg"
    })
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
    security_group_id = aws_security_group.web.id
    cidr_ipv4 = var.allowed_ssh_cidr
    from_port = 22
    to_port = 22
    ip_protocol = "tcp"
    tags = merge(var.tags, {
        Name = "${var.name_prefix}-web-sg"
    })
}

resource "aws_vpc_security_group_ingress_rule" "http" {
    security_group_id = aws_security_group.web.id
    cidr_ipv4 = "0.0.0.0/0"    
    from_port = var.http_port
    to_port = var.http_port
    ip_protocol = "tcp"
    tags = merge(var.tags, {
        Name = "${var.name_prefix}-web-sg"
    })
}


resource "aws_vpc_security_group_egress_rule" "all" {
    security_group_id = aws_security_group.web.id
    cidr_ipv4 = "0.0.0.0/0" 
    ip_protocol = "-1"
    tags = merge(var.tags, {
        Name = "${var.name_prefix}-web-sg"
    })
}