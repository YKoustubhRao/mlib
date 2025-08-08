#!/usr/bin/env bash
set -e  # stop on error
set -u  # error on unset vars



for file in examples/*.py; do
    echo "🔹 Running: $file"
    python "$file"
    echo "✅ Completed: $file"
    echo
done

echo "🎉 All example scripts ran successfully!"
