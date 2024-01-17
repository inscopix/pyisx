if [ -z "$(git status --porcelain)" ]; then 
  # Working directory clean
  echo "✅ setup.py appears to be OK"
else 
  # Uncommitted changes
  git status
  echo "❌ Something has changed; will be safe and raise an error."
  exit 1;
fi
