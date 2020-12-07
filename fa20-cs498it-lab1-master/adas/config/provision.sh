#!/bin/bash
IMAGE_LOC=temp/rasp_image
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "BASE IMAGE INSTALLATION TOOL"

if [ -f "$IMAGE_LOC" ]; then
    echo "Base image found"
else
    echo "Base image not found"
    mkdir -p temp
    echo "Downloading image to temp/rasp_image"
    wget -O $IMAGE_LOC https://downloads.raspberrypi.org/raspios_full_armhf_latest
fi
echo "Verifying download integrity"
echo "24342f3668f590d368ce9d57322c401cf2e57f3ca969c88cf9f4df238aaec41f temp/rasp_image" |sha256sum --check
echo
echo
echo -e "${RED}Warning! The next steps flash an image${NC}"
echo "Incorrect selection may result in TOTAL DATA LOSS"
echo "Confirm no SD card is plugged into the system"
read -rp "Press enter to continue"
BEFORE_DEV=$(awk '{print $4}' < /proc/partitions | sort)
echo "Insert SD card"
read -rp "Press enter to continue"
AFTER_DEV=$(awk '{print $4}' < /proc/partitions | sort)
DIFF=$(comm -13 <(echo "$BEFORE_DEV") <(echo "$AFTER_DEV"))
RAW_DEVICE=$(head -n 1 <(echo "$DIFF"))
DEVICE_BLOCK_SIZE=$(cat "$(echo "/sys/class/block/""$RAW_DEVICE""/size")")
DEVICE=/dev/"$RAW_DEVICE"
if [ "$DEVICE" == "/dev/" ]; then
    echo "No suggested device. Aborting"
    echo "$RAW_DEVICE"
    exit 1
fi
echo "$DEVICE" "is the suggested device"
echo $(( 512 * DEVICE_BLOCK_SIZE / (1024**3)))"GB"
read -rp "Press enter to continue"
echo "Unzipping image"
unzip -n temp/rasp_image -d temp/img/
mv temp/img/*.img temp/img/image.img
echo "Writing image...this may take some time"
sudo dd bs=4M if=temp/img/image.img of=/dev/sda conv=fsync status=progress
echo "Image written to sd card"

