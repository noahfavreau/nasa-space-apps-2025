#!/usr/bin/env bash
if [-f /tmp/reboot.txt]; then
    systemctl restart gunicorn_backend.service
    rm /tmp/reboot.txt
fi

