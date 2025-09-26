# ArgoCD Installation Script for Rap Analyzer - Windows PowerShell
# Phase 2: GitOps Integration

param(
    [switch]$WhatIf
)

Write-Host "🚀 Installing ArgoCD for Rap Analyzer GitOps..." -ForegroundColor Green

# Check if kubectl is available
try {
    kubectl version --client --short | Out-Null
    Write-Host "✅ kubectl found" -ForegroundColor Green
} catch {
    Write-Host "❌ kubectl not found. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Check if we can connect to Kubernetes
try {
    kubectl cluster-info | Out-Null
    Write-Host "✅ Kubernetes connection verified" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to Kubernetes cluster. Please check your kubeconfig." -ForegroundColor Red
    exit 1
}

if ($WhatIf) {
    Write-Host "🔍 WhatIf mode - showing what would be installed..." -ForegroundColor Yellow
    Write-Host "1. ArgoCD namespace and RBAC"
    Write-Host "2. ArgoCD ConfigMaps"
    Write-Host "3. ArgoCD Services and Redis"
    Write-Host "4. ArgoCD Deployments"
    Write-Host "5. Rap Analyzer Application"
    return
}

# Install ArgoCD components
Write-Host "📦 Installing ArgoCD components..." -ForegroundColor Blue

try {
    # Create namespace and RBAC
    Write-Host "   Installing namespace and RBAC..." -ForegroundColor Cyan
    kubectl apply -f gitops/argocd/argocd-install.yaml

    # Wait for namespace to be ready
    Write-Host "⏳ Waiting for argocd namespace..." -ForegroundColor Yellow
    kubectl wait --for=condition=Ready namespace/argocd --timeout=60s

    # Apply ConfigMaps
    Write-Host "   Installing ConfigMaps..." -ForegroundColor Cyan
    kubectl apply -f gitops/argocd/argocd-configmaps.yaml

    # Apply Services and Redis
    Write-Host "   Installing Services and Redis..." -ForegroundColor Cyan
    kubectl apply -f gitops/argocd/argocd-services.yaml

    # Apply Deployments
    Write-Host "   Installing Deployments..." -ForegroundColor Cyan
    kubectl apply -f gitops/argocd/argocd-deployments.yaml

    Write-Host "⏳ Waiting for ArgoCD components to be ready..." -ForegroundColor Yellow

    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s
    kubectl wait --for=condition=available deployment/argocd-repo-server -n argocd --timeout=300s
    kubectl wait --for=condition=available deployment/argocd-application-controller -n argocd --timeout=300s
    kubectl wait --for=condition=available deployment/argocd-redis -n argocd --timeout=300s

    Write-Host "✅ ArgoCD components are ready!" -ForegroundColor Green

    # Apply the rap-analyzer application
    Write-Host "🎵 Deploying rap-analyzer application..." -ForegroundColor Magenta
    kubectl apply -f gitops/applications/rap-analyzer-app.yaml

    Write-Host "🎉 ArgoCD installation complete!" -ForegroundColor Green

} catch {
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Display next steps
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor White
Write-Host "1. Access ArgoCD UI:" -ForegroundColor Cyan
Write-Host "   kubectl port-forward svc/argocd-server -n argocd 8080:443" -ForegroundColor Gray
Write-Host "   Open: https://localhost:8080" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Get admin password:" -ForegroundColor Cyan
Write-Host "   kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Add to C:\Windows\System32\drivers\etc\hosts:" -ForegroundColor Cyan
Write-Host "   127.0.0.1 argocd.rap-analyzer.local" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Check application sync status:" -ForegroundColor Cyan
Write-Host "   kubectl get applications -n argocd" -ForegroundColor Gray
Write-Host ""
Write-Host "🔐 Default credentials:" -ForegroundColor Yellow
Write-Host "   Username: admin" -ForegroundColor Gray
Write-Host "   Password: admin123 (change this!)" -ForegroundColor Gray

# Check deployment status
Write-Host ""
Write-Host "📊 Current Status:" -ForegroundColor White
kubectl get pods -n argocd
Write-Host ""
kubectl get applications -n argocd