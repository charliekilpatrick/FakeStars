#!/bin/bash
# RunAll.sh

obj=$1
./mklists.sh $obj

all_filts=( "g" "r" "i" "z" )

for filt in ${all_filts[@]}; do

	./RunFakeStars.sh $obj $filt > $obj.$filt.out

done	
	
