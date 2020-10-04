#!/bin/bash
fileId=1mO78reKZIVlD9-L6OwuUAQDs_IH-VwFG
fileName=datasets.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
