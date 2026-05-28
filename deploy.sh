#!/bin/bash
# deploy.sh - Deploy catbuddy source files to rasp5.local
# Usage: ./deploy.sh

REMOTE="hrsung@rasp5.local"
REMOTE_DIR="~/work/catbuddy"

echo "Deploying to $REMOTE:$REMOTE_DIR ..."

rsync -avz --exclude='models/' \
           --exclude='*_captures/' \
           --exclude='__pycache__/' \
           --exclude='*.pyc' \
           --exclude='.git/' \
           --exclude='.DS_Store' \
           --exclude='deploy.sh' \
           --exclude='CLAUDE.md' \
           ./ $REMOTE:$REMOTE_DIR/

# Remove stale files that were renamed
ssh $REMOTE "rm -f $REMOTE_DIR/hailo.py"

echo "Done."
