# Security Guidelines

## 🔐 Configuration Security

### Files Already Committed (Safe)

✅ **Safe Files (Templates without secrets):**
- `config.example.yaml` - Configuration template without real credentials
- `.env.example` - Environment variables template with placeholders
- `helm/rap-analyzer/values.yaml` - Helm values with example secrets
- `gitops/argocd/argocd-install.yaml` - Contains `admin123` (example password)
- `gitops/argocd/argocd-configmaps.yaml` - Contains public SSH keys

### 🚨 Files NEVER Committed (Protected by .gitignore)

❌ **Protected Files (Contain real secrets):**
- `.env` - Actual environment variables with API keys
- `.env.local`, `.env.*.local` - Environment-specific secrets
- `config.yaml` - Configuration with real credentials via ENV variables
- `*.production.yaml` - Production-specific overrides
- `*.secret.yaml`, `*-secret.yaml` - Kubernetes secrets

### Configuration Security Model

```yaml
# ✅ SAFE: config.example.yaml (committed to git)
database:
  password_env: "DB_PASSWORD"  # References ENV variable, no actual password

# ✅ SAFE: .env.example (committed to git)
DB_PASSWORD=your_secure_password_here  # Placeholder, not real password

# ❌ SECRET: .env (NEVER commit!)
DB_PASSWORD=actual_real_password_123456  # Real password, in .gitignore

# ❌ SECRET: config.yaml (NEVER commit!)
# Contains same structure but loaded with real ENV variables
```

## 🛡️ Production Security Checklist

### Application Configuration

- [ ] **Copy Templates**: `cp config.example.yaml config.yaml` and `cp .env.example .env`
- [ ] **Fill Real Credentials**: Edit `.env` with actual API keys and passwords
- [ ] **Strong Passwords**: Minimum 16 characters, mixed case, numbers, symbols
- [ ] **API Key Rotation**: Rotate keys every 90 days
- [ ] **Environment Isolation**: Different credentials for dev/staging/prod
- [ ] **2FA Enabled**: On all external services (GitHub, cloud providers)

### Required Environment Variables (See .env.example)

**Critical (Production Required):**
```bash
DB_PASSWORD=<strong-database-password>
NOVITA_API_KEY=<your-qwen-api-key>
```

**Recommended:**
```bash
REDIS_PASSWORD=<redis-password-if-enabled>
GRAFANA_ADMIN_PASSWORD=<strong-grafana-password>
GENIUS_TOKEN=<genius-api-token>
SPOTIFY_CLIENT_ID=<spotify-client-id>
SPOTIFY_CLIENT_SECRET=<spotify-client-secret>
```

## 🔒 Kubernetes & GitOps Security

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