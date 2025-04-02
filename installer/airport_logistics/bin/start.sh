#!/bin/bash  
  
  
# 获取目录路径  
DIRECTORY="./libs/"
  
# 使用find命令查找所有.jar文件，并使用tr命令将换行符替换为:来构建classpath  
# 注意：最后一个:后面需要去掉，所以使用sed命令来删除尾部的:  
CLASSPATH=$(find "$DIRECTORY" -type f -name "*.jar" -print0 | xargs -0 printf "%s:" | sed 's/:$//')

  
# 示例：使用找到的classpath运行一个Java程序（假设主类为com.example.Main）  
# java -cp "$CLASSPATH" com.example.Main
CLASSPATH=./config/:./web_common_demo-1.0.0-SNAPSHOT.jar:./libs/*

# 打印classpath，或者你可以将其用于java命令  
  
echo "Classpath: $CLASSPATH"  
  

java   -Xms1G -Xmx4G  -Djava.library.path=./dll/opencv/ -Dfile.encoding=UTF-8 -cp "$CLASSPATH" airport.cargos.demo.web_common_demo.Bootstrap
