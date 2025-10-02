#!/bin/bash

set -e

echo "🚀 Starting TrafficMetry deployment..."

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull

# Build and deploy containers
echo "🔨 Building containers..."
docker-compose build

echo "🐳 Starting services..."
docker-compose up -d --remove-orphans

# Show status
echo "✅ Deployment completed!"
echo "📊 Service status:"
docker-compose ps

echo "📝 Recent logs:"
docker-compose logs --tail=20