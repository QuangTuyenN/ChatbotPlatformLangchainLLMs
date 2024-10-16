#!/bin/bash
pip list &
uvicorn main_tools:app --host 0.0.0.0 --port 1234