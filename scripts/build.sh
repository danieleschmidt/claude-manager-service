#!/bin/bash
set -e

# Build script for Claude Manager Service
# Usage: ./scripts/build.sh [target] [options]

# Default values
TARGET="${1:-development}"
BUILD_ARGS=""
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-ghcr.io/terragon-labs}"
IMAGE_NAME="${IMAGE_NAME:-claude-manager}"
PUSH="${PUSH:-false}"
CACHE="${CACHE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Claude Manager Service Build Script

Usage: $0 [target] [options]

Targets:
    development    Build development image (default)
    production     Build production image
    testing        Build testing image
    security       Build security scanning image
    all           Build all targets

Options:
    --version=VERSION     Set image version (default: latest)
    --registry=REGISTRY   Set container registry (default: ghcr.io/terragon-labs)
    --push               Push images to registry
    --no-cache           Build without cache
    --help               Show this help message

Environment Variables:
    VERSION              Image version tag
    REGISTRY             Container registry URL
    IMAGE_NAME           Base image name
    PUSH                 Whether to push to registry (true/false)
    CACHE                Whether to use build cache (true/false)

Examples:
    $0                                    # Build development image
    $0 production --version=1.0.0 --push # Build and push production image
    $0 all --no-cache                     # Build all images without cache

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        --registry=*)
            REGISTRY="${1#*=}"
            shift
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --no-cache)
            CACHE="false"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option $1"
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Validate target
case $TARGET in
    development|production|testing|security|all)
        ;;
    *)
        log_error "Invalid target: $TARGET"
        log_info "Valid targets: development, production, testing, security, all"
        exit 1
        ;;
esac

# Set build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

if [ "$CACHE" = "false" ]; then
    BUILD_ARGS="$BUILD_ARGS --no-cache"
fi

BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_DATE=$BUILD_DATE"
BUILD_ARGS="$BUILD_ARGS --build-arg VERSION=$VERSION"
BUILD_ARGS="$BUILD_ARGS --build-arg VCS_REF=$VCS_REF"

# Function to build a specific target
build_target() {
    local target=$1
    local image_tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}-${target}"
    
    log_info "Building $target image: $image_tag"
    
    if docker build \
        $BUILD_ARGS \
        --target "$target" \
        --tag "$image_tag" \
        --tag "${REGISTRY}/${IMAGE_NAME}:latest-${target}" \
        .; then
        log_success "Successfully built $target image"
        
        if [ "$PUSH" = "true" ]; then
            log_info "Pushing $target image to registry..."
            if docker push "$image_tag" && docker push "${REGISTRY}/${IMAGE_NAME}:latest-${target}"; then
                log_success "Successfully pushed $target image"
            else
                log_error "Failed to push $target image"
                return 1
            fi
        fi
    else
        log_error "Failed to build $target image"
        return 1
    fi
}

# Pre-build checks
log_info "Starting build process..."
log_info "Target: $TARGET"
log_info "Version: $VERSION"
log_info "Registry: $REGISTRY"
log_info "Build Date: $BUILD_DATE"
log_info "VCS Ref: $VCS_REF"
log_info "Push to Registry: $PUSH"
log_info "Use Cache: $CACHE"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running or not accessible"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    log_error "Dockerfile not found. Are you in the project root?"
    exit 1
fi

# Create necessary directories
mkdir -p logs data temp backups

# Build target(s)
if [ "$TARGET" = "all" ]; then
    log_info "Building all targets..."
    TARGETS=("development" "production" "testing" "security")
    
    for target in "${TARGETS[@]}"; do
        if ! build_target "$target"; then
            log_error "Build process failed at target: $target"
            exit 1
        fi
    done
    
    log_success "All targets built successfully!"
else
    if ! build_target "$TARGET"; then
        log_error "Build process failed"
        exit 1
    fi
fi

# Generate build report
log_info "Generating build report..."
cat > build-report.json << EOF
{
    "build_date": "$BUILD_DATE",
    "version": "$VERSION",
    "vcs_ref": "$VCS_REF",
    "target": "$TARGET",
    "registry": "$REGISTRY",
    "image_name": "$IMAGE_NAME",
    "pushed": $PUSH
}
EOF

log_success "Build completed successfully!"
log_info "Build report saved to build-report.json"

# Show image sizes
log_info "Image sizes:"
if [ "$TARGET" = "all" ]; then
    for target in development production testing security; do
        docker images "${REGISTRY}/${IMAGE_NAME}:${VERSION}-${target}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
    done
else
    docker images "${REGISTRY}/${IMAGE_NAME}:${VERSION}-${TARGET}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
fi