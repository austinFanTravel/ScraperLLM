#!/bin/bash

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup $BACKUP_FILE..."

# Create backup
tar -czf "$BACKUP_FILE" \
  --exclude='./backups' \
  --exclude='./.git' \
  --exclude='./node_modules' \
  --exclude='./venv' \
  --exclude='./data' \
  .

# Add data directory separately to include it
if [ -d "./data" ]; then
  tar -czf "${BACKUP_FILE}.data.tar.gz" "./data"
  echo "📦 Created data backup: ${BACKUP_FILE}.data.tar.gz"
fi

echo "✅ Backup created: $BACKUP_FILE"
echo "💾 Size: $(du -h "$BACKUP_FILE" | cut -f1)"

# Keep only the last 5 backups
ls -tp "$BACKUP_DIR/" | grep -v '/$' | tail -n +6 | xargs -I {} rm -- "$BACKUP_DIR/{}" 2>/dev/null || true

echo "🔍 Current backups:"
ls -lh "$BACKUP_DIR/" | grep -v "^total" | sort -rh
