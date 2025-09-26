#!/bin/bash

# ArgoCD Installation Script for Rap Analyzer
# Phase 2: GitOps Integration

set -e

echo "🚀 Installing ArgoCD for Rap Analyzer GitOps..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if we can connect to Kubernetes
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

echo "✅ Kubernetes connection verified"

# Apply ArgoCD installation
echo "📦 Installing ArgoCD components..."

# Create namespace and RBAC
kubectl apply -f gitops/argocd/argocd-install.yaml

# Wait for namespace to be ready
echo "⏳ Waiting for argocd namespace..."
kubectl wait --for=condition=Ready namespace/argocd --timeout=60s

# Apply ConfigMaps
kubectl apply -f gitops/argocd/argocd-configmaps.yaml

# Apply Services and Redis
kubectl apply -f gitops/argocd/argocd-services.yaml

# Apply Deployments
kubectl apply -f gitops/argocd/argocd-deployments.yaml

echo "⏳ Waiting for ArgoCD components to be ready..."

# Wait for deployments to be ready
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s
kubectl wait --for=condition=available deployment/argocd-repo-server -n argocd --timeout=300s
kubectl wait --for=condition=available deployment/argocd-application-controller -n argocd --timeout=300s
kubectl wait --for=condition=available deployment/argocd-redis -n argocd --timeout=300s

echo "✅ ArgoCD components are ready!"

# Apply the rap-analyzer application
echo "🎵 Deploying rap-analyzer application..."
kubectl apply -f gitops/applications/rap-analyzer-app.yaml

echo "🎉 ArgoCD installation complete!"

echo ""
echo "📋 Next steps:"
echo "1. Access ArgoCD UI:"
echo "   kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "   Open: https://localhost:8080"
echo ""
echo "2. Get admin password:"
echo "   kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
echo ""
echo "3. Add to /etc/hosts:"
echo "   127.0.0.1 argocd.rap-analyzer.local"
echo ""
echo "4. Check application sync status:"
echo "   kubectl get applications -n argocd"
echo ""
echo "🔐 Default credentials:"
echo "   Username: admin"
echo "   Password: admin123 (change this!)"