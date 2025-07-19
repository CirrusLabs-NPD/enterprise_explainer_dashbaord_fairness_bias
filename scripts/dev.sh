#!/bin/bash

# Development setup script for ML Explainer Dashboard

echo "🚀 Starting ML Explainer Dashboard Development Environment"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or higher is required. Current version: $(node --version)"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Dependencies already installed"
fi

# Check if .env file exists, if not create from example
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# ML Explainer Dashboard Environment Variables
VITE_APP_NAME=ML Explainer Dashboard
VITE_API_URL=http://localhost:8000
VITE_APP_VERSION=1.0.0
VITE_ENABLE_ANALYTICS=false
VITE_ENABLE_DEBUG=true
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Run type checking
echo "🔍 Running type check..."
npm run type-check
if [ $? -ne 0 ]; then
    echo "⚠️  TypeScript errors found, but continuing..."
else
    echo "✅ TypeScript check passed"
fi

# Run linting
echo "🔍 Running linter..."
npm run lint
if [ $? -ne 0 ]; then
    echo "⚠️  Linting errors found, but continuing..."
else
    echo "✅ Linting check passed"
fi

echo ""
echo "🎉 Development environment is ready!"
echo "=================================================="
echo "Available commands:"
echo "  npm run dev          - Start development server"
echo "  npm run build        - Build for production"
echo "  npm run test         - Run tests"
echo "  npm run test:ui      - Run tests with UI"
echo "  npm run lint         - Run linter"
echo "  npm run type-check   - Run TypeScript check"
echo ""
echo "📱 The app will be available at: http://localhost:3000"
echo "🛠️  To start the development server, run: npm run dev"
echo ""

# Ask if user wants to start the dev server
read -p "🚀 Would you like to start the development server now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting development server..."
    npm run dev
fi