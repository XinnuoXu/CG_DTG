FILE_ID=$1
RESOURCE_KEY=$2
TARGET_PATH=$3

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILE_ID}'&resourcekey='${RESOURCE_KEY} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}&resourcekey=${RESOURCE_KEY}" -O ${TARGET_PATH} && rm -rf /tmp/cookies.txt
