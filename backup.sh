#!/bin/bash

# Backup Script for Raspberry Pi Wildlife Camera
# Saves backups to /home/WLCam/Datacube/BACKUPS/[DATE]
# Includes specified files and directories, system configs, logs, SSH keys, package lists, and optional full SD card image.

# Define the backup directory with a date-based folder
BACKUP_DIR="/home/WLCam/Datacube/BACKUPS/WLCam_$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Back up selected Opossum files
echo "Backing up specified Opossum files..."
rsync -av --files-from=<(echo -e ".bashrc\nrclone_credentials.txt\nstatic\ntemplates\nupload_videos.sh\nbrostrend\nOpossum.py\nbackup.sh") /home/WLCam/ "$BACKUP_DIR/home/"

# Backup APT Package List
echo "Backing up APT package list..."
dpkg --get-selections > "$BACKUP_DIR/apt_packages.txt"

# Backup PIP Package List
echo "Backing up PIP package list..."
pip3 freeze > "$BACKUP_DIR/pip_packages.txt"

# Backup Network Configurations (Legacy and NetworkManager)
echo "Backing up network configurations..."

# Backup legacy network configurations if they exist
sudo cp -rL /etc/network/interfaces "$BACKUP_DIR/interfaces" 2>/dev/null
sudo cp -rL /etc/dhcpcd.conf "$BACKUP_DIR/dhcpcd.conf" 2>/dev/null
sudo cp -rL /etc/wpa_supplicant "$BACKUP_DIR/wpa_supplicant" 2>/dev/null

# Backup NetworkManager configurations
nmcli connection show > "$BACKUP_DIR/nmcli_connections.txt"
sudo cp -rL /etc/NetworkManager/system-connections "$BACKUP_DIR/system-connections"
sudo cp -rL /etc/NetworkManager/NetworkManager.conf "$BACKUP_DIR/NetworkManager.conf"

# Backup User and Group Info
echo "Backing up passwd, group, and shadow files..."
sudo cp /etc/passwd "$BACKUP_DIR/passwd"
sudo cp /etc/group "$BACKUP_DIR/group"
sudo cp /etc/shadow "$BACKUP_DIR/shadow"  # Contains hashed passwords. Handle with care.

# Backup SSH Keys and Config
echo "Backing up SSH keys and configuration..."
mkdir -p "$BACKUP_DIR/ssh"
cp -rL ~/.ssh "$BACKUP_DIR/ssh"
sudo cp /etc/ssh/ssh_config "$BACKUP_DIR/ssh/ssh_config"
sudo cp /etc/ssh/sshd_config "$BACKUP_DIR/ssh/sshd_config"

# Backup Boot Configuration
echo "Backing up /boot/firmware/config.txt..."
sudo cp /boot/firmware/config.txt "$BACKUP_DIR/config.txt"

# Optional: Create Full SD Card Image (Commented by Default)
#echo "Creating a full SD card image..."
#SD_CARD_IMAGE="$BACKUP_DIR/sdcard.img"
#sudo dd if=/dev/mmcblk0 of="$SD_CARD_IMAGE" bs=4M status=progress
#sync  # Ensure all writes are flushed to disk

# Step 9: Create a README.md File
echo "Creating README.md file..."
cat <<EOF > "$BACKUP_DIR/README.md"
# Wildlife Camera Backup

This directory contains backups for the Raspberry Pi Wildlife Camera setup. Below is a description of what is included and instructions for restoring the system.

## Backup Contents
- **Opossum Files**:
  - .bashrc
  - rclone_credentials.txt
  - static/
  - templates/
  - upload_videos.sh
  - brostrend/
  - Opossum.py
  - backup.sh
- **apt_packages.txt**: List of installed APT packages.
- **pip_packages.txt**: List of installed PIP packages.
- **Network Configurations**:
  - Legacy: /etc/network/interfaces, /etc/dhcpcd.conf, /etc/wpa_supplicant
  - NetworkManager: /etc/NetworkManager/system-connections, NetworkManager.conf
- **System Files**:
  - /etc/passwd
  - /etc/group
  - /etc/shadow
- **SSH Configuration**:
  - Keys from ~/.ssh
  - /etc/ssh/ssh_config
  - /etc/ssh/sshd_config
- **Boot Configuration**: /boot/firmware/config.txt.
- **SD Card Image**: Full disk image of the SD card as \`sdcard.img\` (if enabled).

## Restore Instructions

### Restore the SD Card
1. Insert a new SD card into your computer.
2. Use the following command to write the backup image to the SD card:
   ```bash
   sudo dd if=sdcard.img of=/dev/mmcblk0 bs=4M status=progress
   sync
