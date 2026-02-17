#!/bin/bash
curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 || exit 1
