#!/usr/bin/env bash
cd /home/serge/prog/nasa-space-apps-2025/src/backend
rm /home/serge/.gunicorn.pid
git stash
git stash drop
git pull


echo "starting Gunicorn"
/home/serge/prog/nasa-space-apps-2025/.venv/bin/gunicorn -w 4 -b 0.0.0.0:8080  wsgi:app --error-logfile /home/serge/error.log --access-logfile /home/serge/access.log --capture-output --log-level debug --pid /home/serge/.gunicorn.pid