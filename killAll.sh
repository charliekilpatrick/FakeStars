#!/bin/bash

for i in `seq 1 10`; do
	for pid in `ps aux | grep $USER | grep run | awk '{print $2}'`; do
		kill -9 $pid
	done
	for pid in `ps aux | grep $USER | grep python | awk '{print $2}'`; do
                kill -9 $pid
        done
done
