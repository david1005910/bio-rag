#!/bin/bash
# Bio-RAG Deployment Script

set -e

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${AWS_REGION:-ap-northeast-2}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG=${2:-latest}

echo "=== Bio-RAG Deployment ==="
echo "Environment: ${ENVIRONMENT}"
echo "Region: ${AWS_REGION}"
echo "Image Tag: ${IMAGE_TAG}"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

# Build and push images
echo "Building and pushing backend image..."
docker build -t bio-rag/backend:${IMAGE_TAG} -f backend/Dockerfile backend/
docker tag bio-rag/backend:${IMAGE_TAG} ${ECR_REPOSITORY}/bio-rag/backend:${IMAGE_TAG}
docker push ${ECR_REPOSITORY}/bio-rag/backend:${IMAGE_TAG}

echo "Building and pushing frontend image..."
docker build -t bio-rag/frontend:${IMAGE_TAG} -f frontend/Dockerfile frontend/
docker tag bio-rag/frontend:${IMAGE_TAG} ${ECR_REPOSITORY}/bio-rag/frontend:${IMAGE_TAG}
docker push ${ECR_REPOSITORY}/bio-rag/frontend:${IMAGE_TAG}

# Update kubeconfig
echo "Updating kubeconfig..."
aws eks update-kubeconfig --region ${AWS_REGION} --name bio-rag-${ENVIRONMENT}

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f infra/kubernetes/namespace.yaml

# Replace variables in manifests
export ECR_REPOSITORY=${ECR_REPOSITORY}
export IMAGE_TAG=${IMAGE_TAG}

envsubst < infra/kubernetes/backend-deployment.yaml | kubectl apply -f -
envsubst < infra/kubernetes/frontend-deployment.yaml | kubectl apply -f -
envsubst < infra/kubernetes/worker-deployment.yaml | kubectl apply -f -
envsubst < infra/kubernetes/ingress.yaml | kubectl apply -f -
kubectl apply -f infra/kubernetes/hpa.yaml

# Wait for rollout
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/bio-rag-backend -n bio-rag --timeout=300s
kubectl rollout status deployment/bio-rag-frontend -n bio-rag --timeout=300s
kubectl rollout status deployment/bio-rag-worker -n bio-rag --timeout=300s

echo "=== Deployment Complete ==="
kubectl get pods -n bio-rag
