#!/bin/bash

for f in $(find . -name '*.jpg' -print)
do
	echo "[i] File: > $f"
	convert $f -sampling-factor 4:2:0 -strip -quality 85 -interlace JPEG -colorspace RGB $f
done

