#/bin/bash
# mklists.sh

if [[ ! -d "lists" ]]; then
	mkdir lists
fi

workdir="/lustre/hpc/storage/dark/YSE/data/workspace"
obj=$1

for dir in `ls -d $workdir/$obj/*/`; do
	basedir=`basename $dir`

	if [ "$basedir" = "tmpl" ]; then
		continue
	fi

	rm lists/$obj.$basedir
	for file in `ls $workdir/$obj/$basedir/*.sw.fits`; do
		basefile=`basename $file`
		echo "$basefile" >> lists/$obj.$basedir
	done
done

for dir in `ls -d $workdir/$obj/tmpl/*/`; do
	basedir=`basename $dir`

	rm lists/$obj.$basedir.tmpl
	for file in `ls $workdir/$obj/tmpl/$basedir/*.sw.fits`; do
                basefile=`basename $file`
                echo "$basefile" >> lists/$obj.$basedir.tmpl
        done
done
