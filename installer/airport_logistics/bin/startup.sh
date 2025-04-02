#!/bin/bash
export JAVA_HOME=/usr/local/jdk
export PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:${JAVA_HOME}/bin

        #定义变量
        PROJECT_PATH=$(cd $(dirname "$0");pwd)
        PROJECT_NAME=`echo $PROJECT_PATH|awk -F '/' '{print $NF}'`
        LOG_PATH="/data/logs/cargos/${PROJECT_NAME}"
        #将日志路径加载到环境变量
        export LOG_PATH
        mkdir -p $LOG_PATH
        local_ip=`ifconfig eth0|head -2|grep inet|awk '{print $2}'`
        PID_FILE="${PROJECT_PATH}/${PROJECT_NAME}.pid"
    if [ -f "$PID_FILE" ]; then rm -f $PID_FILE ;fi
        JAVA_DEBUG_OPT=" -Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=550,suspend=n"

        cd $PROJECT_PATH

    #定义服务域
    DOMAIN="airport.cargos.demo.Bootstrap"
    SERVER_CLASS="${DOMAIN}"
    co=`ps -ef | grep "${SERVER_CLASS}" | grep -v "grep" | wc -l`

    if [ ${co} -eq 0 ];then
        libs="${PROJECT_PATH}/conf`ls -1 ${PROJECT_PATH}/lib/*.jar | awk '{printf ":"$1}'`"
        #加载lib包
        export CLASSPATH=$CLASSPATH:$libs
        #程序启动
###########################这一段自己更换一下########################
        java \
                                -server \
                                -Djava.net.preferIPv4Stack=true \
                                -Xloggc:gc.log \
                                -XX:+PrintReferenceGC \
                                -XX:+ParallelRefProcEnabled \
                                -Xmx256M \
                                -Xms256M \
                                -Xss256k \
                                -XX:PermSize=128M \
                                -XX:MaxPermSize=128M \
                                -Dlocal.ip=$local_ip \
                                -Dproject.path=$PROJECT_PATH \
                                -XX:+PrintGCDateStamps \
                                -XX:+PrintGCDetails \
                                -XX:+UseG1GC \
                                -XX:+UnlockExperimentalVMOptions \
                                -XX:G1LogLevel=finest \
                                -XX:InitiatingHeapOccupancyPercent=25 \
                                -XX:MaxGCPauseMillis=200 \
                                -XX:G1HeapRegionSize=4m \
                                -XX:+PrintFlagsFinal \
                                -XX:ParallelGCThreads=12 \
                                -XX:ConcGCThreads=4 \
                                ${SERVER_CLASS}  1>${LOG_PATH}/server_stdout_log.$(date +%s) 2>&1  &
########################################################################
        sleep 1
        echo `ps -ef|grep "$$"|grep "${DOMAIN}"|grep -v grep|awk '{print $2}'` > ${PID_FILE}
        echo "Server is running"
    else
        echo "Start failed,server is already running!"
    fi
