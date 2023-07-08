#!/bin/bash
#for fil in $1/*.zip; do
#	dst=/sharefs/baai-mmdataset/wudaomm-all/${fil%*.zip}	
#	echo $dst
	#mkdir $dst
	#sudo unzip $fil -d $dst
#done

#for file in $1/*.rar; do
	#echo ${file%*.rar}
#	dst=/sharefs/baai-mmdataset/wudaomm-all/${file%*.rar}
#	echo $dst
	#mkdir $dst	
	#sudo unrar x $file $dst
#done
cd $1
for file in *.tar; do
	#echo ${file%*.rar}
	dst=/sharefs/baai-mmdataset/wudaomm-all/${file%*.tar}
	echo $dst
	mkdir $dst     
	sudo tar -xvf $file -C $dst
done
