#!/bin/bash  
  
# 设置要搜索的目录，默认为当前目录  
search_dir="./libs/"  
  
# 初始化类路径为空  
classpath=""  
  
# 使用find命令查找所有JAR文件，并构建类路径  
# 注意：这里我们假设JAR文件的后缀为.jar，并且不考虑子目录中的JAR文件  
# 如果需要递归搜索子目录，可以将-maxdepth 1去掉  
for jarfile in $(find "$search_dir" -maxdepth 1 -name "*.jar"); do  
    # 检查类路径是否为空，如果不为空则添加冒号分隔符  
    if [[ -n "$classpath" ]]; then  
        classpath="$classpath:"  
    fi  
    # 将JAR文件的绝对路径添加到类路径  
    classpath="$classpath$jarfile"  
done  
  
# 输出类路径或用于其他命令  
echo "Classpath: $classpath"  
  
# 例如，使用构建的类路径运行Java程序  
java -cp ./config/:"$classpath":./web_common_demo-1.0.0-SNAPSHOT.jar airport.cargos.demo.web_common_demo.Bootstrap
