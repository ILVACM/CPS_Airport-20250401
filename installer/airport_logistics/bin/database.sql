
drop database if EXISTS airport_logistics;
drop user if EXISTS  'airport'@'localhost';
set names utf8mb4;
create database airport_logistics;
create user 'airport'@'localhost' identified by 'P@ssw0rd';  
GRANT ALL PRIVILEGES ON airport_logistics.* TO 'airport'@'localhost';

use airport_logistics;

CREATE TABLE `contraband_image` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `time` DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    `place` VARCHAR(255) NOT NULL,
    `name` VARCHAR(100) NOT NULL,
    `view` INT NOT NULL,
    `detected_image_path` VARCHAR(100) NOT NULL,
    `original_image_path` VARCHAR(100) NOT NULL
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4;

CREATE TABLE `user` (
                        `id` int NOT NULL AUTO_INCREMENT,
                        `role` varchar(32) NOT NULL,
                        `username` varchar(32) NOT NULL,
                        `password` varchar(32) NOT NULL,
                        `realName` varchar(45) NOT NULL DEFAULT ' ',
                        `create_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (`id`),
                        UNIQUE KEY `id_UNIQUE` (`id`),
                        UNIQUE KEY `username_UNIQUE` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4 ;

insert into user(role,username,password,realName)values('1','admin','123456','管理员');

CREATE TABLE `videopath` (
                             `id` int NOT NULL AUTO_INCREMENT,
                             `path` varchar(200) NOT NULL,
                             `name` varchar(255) NOT NULL,
                             `upload_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
                             `description` varchar(45) DEFAULT NULL,
                             `user` varchar(50) DEFAULT NULL,
                             PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `contraband_catrgorys`;
CREATE TABLE `contraband_catrgorys`  (
                                         `id` int NOT NULL AUTO_INCREMENT,
                                         `categoryName` varchar(128) CHARACTER SET utf8mb4 NOT NULL,
                                         `name` varchar(128) CHARACTER SET utf8mb4 NOT NULL,
                                         `status` tinyint(1) NOT NULL,
                                         PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 26 CHARACTER SET = utf8mb4  ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of contraband_catrgorys
-- ----------------------------
INSERT INTO `contraband_catrgorys` VALUES (1, '管制器具', '电击器', 1);
INSERT INTO `contraband_catrgorys` VALUES (2, '管制器具', '警棍', 1);
INSERT INTO `contraband_catrgorys` VALUES (3, '枪支', '枪支', 1);
INSERT INTO `contraband_catrgorys` VALUES (4, '管制器具', '手铐', 1);
INSERT INTO `contraband_catrgorys` VALUES (5, '管制器具', '管制刀具', 1);
INSERT INTO `contraband_catrgorys` VALUES (6, '烟火制品', '鞭炮', 1);
INSERT INTO `contraband_catrgorys` VALUES (7, '烟火制品', '电子烟花', 1);
INSERT INTO `contraband_catrgorys` VALUES (8, '烟火制品', '礼花', 1);
INSERT INTO `contraband_catrgorys` VALUES (9, '烟火制品', '烟花', 1);
INSERT INTO `contraband_catrgorys` VALUES (10, '烟火制品', '烟饼', 1);
INSERT INTO `contraband_catrgorys` VALUES (11, '烟火制品', '仙女棒', 1);
INSERT INTO `contraband_catrgorys` VALUES (12, '危险物品', '压缩管', 1);
INSERT INTO `contraband_catrgorys` VALUES (13, '弹药类', '子弹', 1);
INSERT INTO `contraband_catrgorys` VALUES (14, '弹药类', '空包弹', 1);
INSERT INTO `contraband_catrgorys` VALUES (15, '弹药类', '霰弹', 1);
INSERT INTO `contraband_catrgorys` VALUES (16, '弹药类', '烟雾弹', 1);
INSERT INTO `contraband_catrgorys` VALUES (17, '火种', '塑料打火机', 1);
INSERT INTO `contraband_catrgorys` VALUES (18, '火种', '金属打火机', 1);
INSERT INTO `contraband_catrgorys` VALUES (19, '火种', '镁棒点火器', 1);
INSERT INTO `contraband_catrgorys` VALUES (20, '火种', '防风火柴', 1);
INSERT INTO `contraband_catrgorys` VALUES (21, '电池类', '块状电池', 1);
INSERT INTO `contraband_catrgorys` VALUES (22, '电池类', '节状电池', 1);
INSERT INTO `contraband_catrgorys` VALUES (23, '电池类', '纽扣电池', 1);
INSERT INTO `contraband_catrgorys` VALUES (24, '电池类', '蓄电池', 1);
INSERT INTO `contraband_catrgorys` VALUES (25, '电池类', '蓝牙耳机', 1);
SET FOREIGN_KEY_CHECKS = 1;


--  insert into contraband_image(place,name,view,detected_image_path) values("白云机场","小刀",0,"D:\\Codes\\机场物流项目代码\\yolo_tracking-8.0\\images\\2916_down.jpg");
