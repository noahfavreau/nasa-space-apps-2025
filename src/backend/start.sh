#!/usr/bin/env bash
cd /home/serge/prog/nasa-space-apps-2025/src/backend
git stash
git stash drop
git pull

/home/serge/prog/nasa-space-apps-2025/.venv/bin/gunicorn -w 4 -b 0.0.0.0:8080  wsgi:app --error-logfile error.log --access-logfile access.log --capture-output --log-level debug --pid /run/gunicorn.pid