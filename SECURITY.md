# Kubernetes & GitOps Security Guidelines

## Files Already Committed (Safe)

The following files contain only **development/example** credentials and are safe for public repositories:

### ✅ Safe Files (Example/Development Credentials):
- `gitops/argocd/argocd-install.yaml` - Contains `admin123` (example password)
- `gitops/argocd/argocd-configmaps.yaml` - Contains public SSH keys
- `helm/rap-analyzer/values.yaml` - Contains example secrets

### 🔒 Production Security

For **production deployments**, create these files locally (NOT committed to Git):

#### 1. Production ArgoCD Secret
```bash
# Create file: gitops/argocd/argocd-secret.production.yaml
apiVersion: v1
kind: Secret
metadata:
  name: argocd-secret
  namespace: argocd
type: Opaque
data:
  admin.password: <your-secure-base64-encoded-password>
  admin.passwordMtime: <current-timestamp-base64>
```

#### 2. Production Helm Values
```bash
# Create file: helm/rap-analyzer/values.production.yaml
security:
  secrets:
    postgresql:
      password: "your-production-db-password"
    api:
      secretKey: "your-production-api-secret"
      openaiApiKey: "your-production-openai-key"
```

#### 3. Production Deployment Commands
```bash
# ArgoCD with production secrets
kubectl apply -f gitops/argocd/argocd-secret.production.yaml

# Helm with production values
helm install rap-analyzer ./helm/rap-analyzer \
  -f helm/rap-analyzer/values.yaml \
  -f helm/rap-analyzer/values.production.yaml \
  --namespace rap-analyzer
```

## .gitignore Protection

The following patterns are now excluded from Git:

```
# Kubernetes secrets and sensitive data
k8s/secrets/
gitops/secrets/
*.secret.yaml
*-secret.yaml
*.production.yaml

# ArgoCD generated files
gitops/argocd/argocd-initial-admin-secret.yaml
gitops/.argocd/

# TLS certificates and keys
*.crt
*.key
*.pem
*.p12
tls/

# Registry credentials
.dockerconfigjson
docker-config.json
```

## Best Practices

1. **Development**: Use example credentials (already committed)
2. **Production**: Create `.production.yaml` files locally
3. **CI/CD**: Use Kubernetes secrets or external secret management
4. **Team Sharing**: Use secure channels for production credentials

## Current Status: ✅ SAFE

All committed files contain only development/example credentials. No real secrets were exposed.