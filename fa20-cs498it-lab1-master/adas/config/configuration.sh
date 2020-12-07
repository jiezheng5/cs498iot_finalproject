#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color
MOUNT_POINT=/mnt/rpihome
MOUNT_BOOT=/mnt/bootmount
source env.sh

echo "BASE IMAGE CONFIGURATION"
echo "Please enter the boot location (likely /dev/sda1)"
read DEVICE

echo "Mounting boot"
mount "$DEVICE" "$MOUNT_BOOT"


echo "BASE IMAGE CONFIGURATION"
echo "Please enter the configuration destination (likely /dev/sda2)"
read DEVICE

echo "Mounting image"
mount "$DEVICE" "$MOUNT_POINT"


echo "Enabling ssh"
touch "$MOUNT_BOOT"/ssh


echo "Configuring hostname"
echo "$ADAS_HOSTNAME" > "$MOUNT_POINT"/etc/hostname

echo "Configure static address for ethernet"
cat <<EOF > "$MOUNT_POINT"/etc/dhcpcd.conf
interface eth0
static ip_address=$ADAS_IP$ADAS_SUBNET
EOF


echo "Configuring wireless"
cat <<EOF > "$MOUNT_BOOT"/wpa_supplicant.conf
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
ssid="$WIRELESS_NETWORK"
scan_ssid=1
psk="$WIRELESS_PASSWORD"
key_mgmt=WPA-PSK
}
EOF

umount "$MOUNT_POINT"
umount "$MOUNT_BOOT"
