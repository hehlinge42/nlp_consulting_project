#!/bin/bash

input="./rooftops.txt"
while IFS= read -r line
do
	echo "$line"
done < "$input"
