#!/usr/bin/env bash
git fetch origin
A= $(git log -1 --format=%ci origin/$(git rev-parse --abbrev-ref HEAD))
Current= $(cat current.txt)
if [$((A-current)) > 0]; then
    systemctl restart gunicorn_backend.service
    echo $A > current.txt
fi

