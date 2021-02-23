#!/bin/bash

i=-2
input="./rooftops.txt"
while IFS= read -r url
do
	if [[ ${url:0:1} == "#" ]];
	then
		echo SKIPPING COMMENT;
	else
		echo LAUNCHING SPIDER
		scrapy crawl BokanSpider -a debug=1 -a root_url=$url -a id_resto=$i
		i=$((i-1));
	fi
done < "$input"
