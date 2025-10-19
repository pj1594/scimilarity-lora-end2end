#!/usr/bin/env bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"expression":{"CD19":8.5,"MS4A1":9.2,"CD79A":7.8}}'
