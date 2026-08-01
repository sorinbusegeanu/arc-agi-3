#!/bin/bash

# Define variables
SWAP_PATH="/swapfile"
SWAP_SIZE_GB=128

echo "Creating a ${SWAP_SIZE_GB}GB swap file at ${SWAP_PATH}..."

# 1. Allocate space safely using fallocate
sudo fallocate -l ${SWAP_SIZE_GB}G $SWAP_PATH

# Fallback to dd if fallocate is not supported by the filesystem (e.g., XFS older versions)
if [ $? -ne 0 ]; then
    echo "fallocate failed, falling back to dd (this may take a few minutes)..."
    sudo dd if=/dev/zero of=$SWAP_PATH bs=1M count=$((SWAP_SIZE_GB * 1024))
fi

# 2. Set strict permissions (security requirement)
sudo chmod 600 $SWAP_PATH

# 3. Set up the Linux swap area
sudo mkswap $SWAP_PATH

# 4. Enable the swap file immediately
sudo swapon $SWAP_PATH

# 5. Make the swap file permanent across reboots
if ! grep -q "$SWAP_PATH" /etc/fstab; then
    echo "$SWAP_PATH none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "Added swap entry to /etc/fstab."
else
    echo "Swap entry already exists in /etc/fstab."
fi

echo "64GB Swap file successfully created and activated!"
free -h
