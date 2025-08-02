#!/bin/bash
# Manual Railway Setup - Run after creating service in dashboard

echo "🔗 Linking to Railway service..."
echo "Select your 'cpuc-rag-system-web' service from the menu:"
railway service

echo ""
echo "⚙️  Setting environment variables..."

railway variables --set "USE_CLOUD_STORAGE=true"
railway variables --set "CLOUD_STORAGE_TYPE=s3"
railway variables --set "S3_BUCKET=cpuc-rag-vectors-prod"  
railway variables --set "AWS_DEFAULT_REGION=us-west-2"
railway variables --set "PYTHONPATH=/app/src"
railway variables --set "MAX_CHUNKS_PER_DOCUMENT=1500"
railway variables --set "STREAMLIT_SERVER_PORT=8080"
railway variables --set "STREAMLIT_SERVER_ADDRESS=0.0.0.0"
railway variables --set "STREAMLIT_SERVER_HEADLESS=true"

echo ""
echo "🔐 Please set these sensitive variables in Railway dashboard:"
echo "   - AWS_ACCESS_KEY_ID (your AWS access key)"
echo "   - AWS_SECRET_ACCESS_KEY (your AWS secret key)"  
echo "   - OPENAI_API_KEY (your OpenAI API key)"
echo ""
echo "Railway Dashboard: https://railway.app/dashboard"
echo ""
echo "🚀 Deploy with: railway up"