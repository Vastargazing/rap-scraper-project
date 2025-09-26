# GitOps with ArgoCD - Phase 2

## Overview

This directory contains ArgoCD configurations for implementing GitOps practices in the Rap Analyzer project. ArgoCD provides automated deployment, configuration management, and continuous delivery capabilities.

## Architecture

```
┌─────────────────── GITOPS WORKFLOW ───────────────────┐
│                                                      │
│  ┌─── DEVELOPER ────┐    ┌─── GIT REPOSITORY ────┐   │
│  │ • Code Changes   │───▶│ • Helm Charts        │   │
│  │ • Config Updates │    │ • K8s Manifests      │   │
│  │ • Git Push       │    │ • Configuration      │   │
│  └──────────────────┘    └───────────┬──────────┘   │
│                                      │              │
│                                      ▼              │
│  ┌─── ARGOCD CONTROLLER ──────────────────────────┐  │
│  │ • Monitors Git Repository                     │  │
│  │ • Detects Configuration Changes               │  │
│  │ • Syncs Desired State to Cluster              │  │
│  │ • Provides UI and CLI for Management          │  │
│  └───────────────────────┬────────────────────────┘  │
│                          │                          │
│                          ▼                          │
│  ┌─── KUBERNETES CLUSTER ─────────────────────────┐  │
│  │ • Rap Analyzer Application                    │  │
│  │ • PostgreSQL Database                         │  │
│  │ • Monitoring Stack                            │  │
│  │ • Auto-scaling and Health Checks              │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Components

### ArgoCD Installation

- **argocd-install.yaml** - Namespace, RBAC, and basic configuration
- **argocd-deployments.yaml** - Server, Repo Server, Application Controller
- **argocd-services.yaml** - Services, Redis, and Ingress
- **argocd-configmaps.yaml** - SSH keys, TLS certs, and parameters

### Applications

- **rap-analyzer-app.yaml** - Main application configuration with:
  - Automated sync policy
  - Self-healing capabilities
  - Multi-environment support
  - Production-optimized settings

## Installation

### Prerequisites

1. **Kubernetes Cluster** (v1.20+)
2. **kubectl** configured
3. **Ingress Controller** (nginx recommended)
4. **Helm 3.0+**

### Quick Install

```bash
# Windows PowerShell
./gitops/install-argocd.ps1

# Linux/Mac
chmod +x gitops/install-argocd.sh
./gitops/install-argocd.sh
```

### Manual Installation

```bash
# 1. Install ArgoCD components
kubectl apply -f gitops/argocd/argocd-install.yaml
kubectl apply -f gitops/argocd/argocd-configmaps.yaml
kubectl apply -f gitops/argocd/argocd-services.yaml
kubectl apply -f gitops/argocd/argocd-deployments.yaml

# 2. Wait for components to be ready
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s

# 3. Deploy rap-analyzer application
kubectl apply -f gitops/applications/rap-analyzer-app.yaml
```

## Access ArgoCD

### Web UI

```bash
# Port forward to access UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Open browser
https://localhost:8080
```

### Credentials

```bash
# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d

# Default: admin/admin123 (change immediately!)
```

### CLI

```bash
# Install ArgoCD CLI
curl -sSL -o argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
chmod +x argocd
sudo mv argocd /usr/local/bin/

# Login
argocd login localhost:8080

# List applications
argocd app list
```

## Application Management

### Sync Application

```bash
# Manual sync
kubectl patch application rap-analyzer -n argocd --type merge --patch '{"operation":{"sync":{}}}'

# Or via CLI
argocd app sync rap-analyzer
```

### Check Status

```bash
# Get application status
kubectl get application rap-analyzer -n argocd -o yaml

# Check sync status
argocd app get rap-analyzer
```

### Rollback

```bash
# List revisions
argocd app history rap-analyzer

# Rollback to previous version
argocd app rollback rap-analyzer <revision-id>
```

## Environment Management

### Development Environment

Create `rap-analyzer-dev-app.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rap-analyzer-dev
  namespace: argocd
spec:
  source:
    repoURL: 'https://github.com/Vastargazing/rap-scraper-project'
    targetRevision: develop
    path: helm/rap-analyzer
    helm:
      values: |
        app:
          api:
            replicaCount: 1
        monitoring:
          enabled: false
  destination:
    namespace: rap-analyzer-dev
```

### Staging Environment

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rap-analyzer-staging
  namespace: argocd
spec:
  source:
    repoURL: 'https://github.com/Vastargazing/rap-scraper-project'
    targetRevision: staging
    path: helm/rap-analyzer
    helm:
      values: |
        app:
          api:
            replicaCount: 2
        postgresql:
          primary:
            persistence:
              size: 50Gi
  destination:
    namespace: rap-analyzer-staging
```

## Security

### RBAC Configuration

ArgoCD includes role-based access control:

- **Admin Role**: Full access to all applications
- **Developer Role**: Read and sync access
- **Viewer Role**: Read-only access

### Repository Access

Configure repository credentials in ArgoCD:

```bash
# Add private repository
argocd repo add https://github.com/your-org/private-repo \
  --username <username> \
  --password <token>
```

### Secret Management

For sensitive data, use:

1. **Kubernetes Secrets**
2. **External Secret Operator**
3. **Sealed Secrets**
4. **Vault Integration**

## Monitoring

### ArgoCD Metrics

ArgoCD exports Prometheus metrics:

- Application sync status
- Repository connection health
- Controller performance metrics

### Integration with Grafana

Add ArgoCD dashboards to existing Grafana:

```bash
# Import dashboard ID: 14584 (ArgoCD)
kubectl patch configmap grafana-dashboard-argocd -n rap-analyzer --patch '{"data":{"argocd-dashboard.json":"..."}}'
```

## Troubleshooting

### Common Issues

1. **Sync Failures**
   ```bash
   kubectl logs deployment/argocd-application-controller -n argocd
   ```

2. **Repository Access**
   ```bash
   kubectl logs deployment/argocd-repo-server -n argocd
   ```

3. **Server Issues**
   ```bash
   kubectl logs deployment/argocd-server -n argocd
   ```

### Debugging Commands

```bash
# Check application health
argocd app get rap-analyzer --show-params

# View sync operation
argocd app sync rap-analyzer --dry-run

# Check repository connection
argocd repo list
```

## Best Practices

1. **Git Workflow**
   - Use feature branches for changes
   - Require pull request reviews
   - Tag releases for production

2. **Application Configuration**
   - Enable automated sync with caution
   - Use self-healing for simple changes
   - Monitor sync failures

3. **Security**
   - Rotate ArgoCD admin password
   - Use RBAC for team access
   - Audit configuration changes

4. **Monitoring**
   - Set up alerts for sync failures
   - Monitor application health
   - Track deployment frequency

## Next Steps

After GitOps setup:

1. **Multi-region Deployment** - Configure ArgoCD for multiple clusters
2. **Advanced Security** - Implement policy-as-code with OPA Gatekeeper
3. **Progressive Delivery** - Add Argo Rollouts for canary deployments
4. **Observability** - Integrate with distributed tracing