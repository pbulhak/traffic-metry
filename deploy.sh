#!/bin/bash
# TrafficMetry Intelligent Deployment Script
# Handles environment setup, model optimization, and deployment

set -e  # Exit on error
set -u  # Exit on undefined variable

# === CONFIGURATION ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
ENV_EXAMPLE="${SCRIPT_DIR}/.env.example"
SOURCE_MODEL_PATH="data/models/yolov8n.pt"
OPENVINO_MODEL_PATH="data/models/yolov8n_int8_openvino_model"
OPENVINO_MARKER="${OPENVINO_MODEL_PATH}/.conversion_complete"

# === HELPER FUNCTIONS ===
log_info() { echo "ℹ️  $1"; }
log_success() { echo "✅ $1"; }
log_warning() { echo "⚠️  $1"; }
log_error() { echo "❌ $1" >&2; }

# === STEP 1: Environment File Management ===
setup_env_file() {
    log_info "Checking .env file..."

    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found, creating from .env.example..."

        if [ ! -f "$ENV_EXAMPLE" ]; then
            log_error ".env.example not found! Cannot proceed."
            exit 1
        fi

        cp "$ENV_EXAMPLE" "$ENV_FILE"
        log_success ".env created from template"
        log_warning "IMPORTANT: Please edit .env with your actual configuration!"
        log_info "Press Enter when ready to continue, or Ctrl+C to abort..."
        read -r
    else
        log_success ".env file exists"
    fi
}

# === STEP 2: Model Optimization ===
optimize_model() {
    log_info "Checking model optimization status..."

    # Check if source model exists
    if [ ! -f "$SOURCE_MODEL_PATH" ]; then
        log_warning "Source model not found at $SOURCE_MODEL_PATH"
        log_info "Will use default model from Ultralytics"
        return 0
    fi

    # Check if OpenVINO model already exists and is fresh
    if [ -f "$OPENVINO_MARKER" ]; then
        log_success "OpenVINO model already optimized (found marker)"
        ensure_env_uses_openvino
        return 0
    fi

    log_info "🔄 Starting model conversion to OpenVINO INT8..."
    log_info "This may take 2-5 minutes on first run..."

    # Run model export using docker-compose
    docker-compose run --rm model-exporter \
        uv run python scripts/export_model.py \
        --source "$SOURCE_MODEL_PATH"

    # Create marker file to indicate successful conversion
    mkdir -p "$(dirname "$OPENVINO_MARKER")"
    date > "$OPENVINO_MARKER"

    log_success "Model conversion completed!"

    # Update .env to use optimized model
    ensure_env_uses_openvino
}

# === STEP 3: Update .env to use OpenVINO ===
ensure_env_uses_openvino() {
    log_info "Updating .env to use optimized OpenVINO model..."

    if grep -q "^MODEL__PATH=\"${OPENVINO_MODEL_PATH}\"" "$ENV_FILE"; then
        log_success ".env already configured for OpenVINO"
        return 0
    fi

    # Backup original .env
    cp "$ENV_FILE" "${ENV_FILE}.backup"

    # Update MODEL__PATH to point to OpenVINO
    if grep -q "^MODEL__PATH=" "$ENV_FILE"; then
        sed -i "s|^MODEL__PATH=.*|MODEL__PATH=\"${OPENVINO_MODEL_PATH}\"|" "$ENV_FILE"
    else
        echo "MODEL__PATH=\"${OPENVINO_MODEL_PATH}\"" >> "$ENV_FILE"
    fi

    log_success ".env updated to use OpenVINO model"
    log_info "Backup saved to ${ENV_FILE}.backup"
}

# === STEP 4: Git Operations ===
update_code() {
    log_info "📥 Pulling latest changes..."

    # Check if there are uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_warning "Uncommitted changes detected"
        log_info "Stashing changes..."
        git stash
    fi

    git pull
    log_success "Code updated"
}

# === STEP 5: Docker Deployment ===
deploy_containers() {
    log_info "🔨 Building containers..."
    docker-compose build

    log_info "🐳 Starting services..."
    docker-compose up -d --remove-orphans

    log_success "Deployment completed!"
}

# === STEP 6: Health Check ===
show_status() {
    log_info "📊 Service status:"
    docker-compose ps

    echo ""
    log_info "📝 Recent logs:"
    docker-compose logs --tail=20

    echo ""
    log_info "🔍 Model configuration:"
    grep "^MODEL__PATH=" "$ENV_FILE" || log_warning "MODEL__PATH not set in .env"
}

# === MAIN EXECUTION ===
main() {
    echo "🚀 TrafficMetry Intelligent Deployment"
    echo "========================================"
    echo ""

    setup_env_file
    optimize_model
    update_code
    deploy_containers
    show_status

    echo ""
    log_success "🎉 Deployment complete!"
    log_info "Access the application at http://localhost"
}

# Run main function
main "$@"
