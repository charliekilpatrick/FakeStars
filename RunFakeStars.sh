#!/bin/bash

root_path=$PIPE_DATA
log_base="logs"
work_base="workspace"
image_dir=$1
template_dir="tmpl"
filter=$2
image_list="lists/${image_dir}.${filter}"
tmpl_list="lists/${image_dir}.${filter}.tmpl"
coord_list="coords/${image_dir}_coord_list.txt"
fake_bright="18"
fake_dim="24"
num_stars=1500
target_efficiency="0.8"

start=$SECONDS

msg="Processing uniform fakes between ${fake_bright} and ${fake_dim}"
echo $msg
#
# Do work...
python ./Fakes.py \
--root_path ${root_path} \
--log_base ${log_base} \
--work_base ${work_base} \
--image_dir ${image_dir} \
--template_dir ${template_dir} \
--image_list ${image_list} \
--template_list ${tmpl_list} \
--coord_list ${coord_list} \
--fake_mag_range ${fake_bright} ${fake_dim} ${num_stars} \
--filter ${filter} \
--target-efficiency ${target_efficiency} \
--save-img \
--use-diffimstats
