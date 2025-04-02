#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
PROJ_PATH=$(cd $(dirname "$0");pwd)
if [ $# -le 0 ];then
    echo "请输入实例名"
    exit
else
    IDX=$1
    PID_FILE="${PROJ_PATH}/${IDX}.pid"
    cd $PROJ_PATH
    if [ -f ${PID_FILE} ];then
        kill `cat ${PID_FILE}`
        count=0
        while true
        do
            sleep 2
            check=$(ps -ef|grep -v grep|awk -v p="$(cat ${PID_FILE})" '{if($2==p) print $0}')
            if [ -z "${check}" ];then
                rm -f ${PID_FILE}
                break
            fi
            let count++
            if [ ${count} -gt 15 ];then
                break
            fi
        done
        #自定义内容#

    else
        echo "No such pid file ${PID_FILE}!"
    fi
fi