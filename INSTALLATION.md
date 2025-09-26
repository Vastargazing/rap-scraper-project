# INSTALLATION.md

## Phase 1: Kubernetes Migration - Installation Guide

### Prerequisites

1. **Kubernetes Cluster** (v1.20+)
   - kubectl configured and connected to cluster
   - Ingress controller (nginx recommended)
   - At least 8GB RAM and 4 CPU cores available

2. **Helm** (v3.0+)
   ```bash
   # Install Helm
   curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
   sudo mv linux-amd64/helm /usr/local/bin/
   ```

3. **Docker Images Built**
   ```bash
   # Build the application image
   docker build -f Dockerfile.k8s -t rap-analyzer/api:1.0.0 .
   
   # Tag for your registry (if using remote registry)
   docker tag rap-analyzer/api:1.0.0 your-registry/rap-analyzer/api:1.0.0
   docker push your-registry/rap-analyzer/api:1.0.0
   ```

### Quick Start

1. **Deploy using Helm Chart**
   ```bash
   # Install the complete stack
   helm install rap-analyzer ./helm/rap-analyzer \
     --create-namespace \
     --namespace rap-analyzer
   ```

2. **Verify Deployment**
   ```bash
   # Check pod status
   kubectl get pods -n rap-analyzer
   
   # Check services
   kubectl get svc -n rap-analyzer
   
   # Check ingress
   kubectl get ingress -n rap-analyzer
   ```

3. **Access Applications**
   ```bash
   # Add to /etc/hosts (or use LoadBalancer IP)
   echo "127.0.0.1 rap-analyzer.local" >> /etc/hosts
   echo "127.0.0.1 grafana.rap-analyzer.local" >> /etc/hosts
   echo "127.0.0.1 prometheus.rap-analyzer.local" >> /etc/hosts
   
   # Forward ports for local testing
   kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
   kubectl port-forward svc/grafana-service 3000:3000 -n rap-analyzer
   kubectl port-forward svc/prometheus-service 9090:9090 -n rap-analyzer
   ```

4. **Test API**
   ```bash
   # Health check
   curl http://rap-analyzer.local/health
   
   # API documentation
   curl http://rap-analyzer.local/docs
   ```

### Alternative: Manual K8s Manifests

If you prefer to deploy without Helm:

```bash
# Deploy in order
kubectl apply -f k8s/namespace-and-config.yaml
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/api/
kubectl apply -f k8s/monitoring/
kubectl apply -f k8s/ingress.yaml
```

### Configuration

#### Database Initialization

The PostgreSQL deployment includes automatic initialization:
- Creates `rap_lyrics` database
- Installs pgvector extension
- Sets up required schemas

#### Secrets Management

Update secrets in `k8s/namespace-and-config.yaml` or Helm values:

```yaml
# In values.yaml
security:
  secrets:
    postgresql:
      password: "your-secure-password"
    api:
      secretKey: "your-api-secret-key"
      openaiApiKey: "your-openai-key"
```

#### Scaling Configuration

```yaml
# In values.yaml
app:
  api:
    replicaCount: 5  # Manual scaling
    autoscaling:
      enabled: true
      minReplicas: 3
      maxReplicas: 20
      targetCPUUtilizationPercentage: 70
```

### Monitoring Setup

#### Grafana Access
- URL: http://grafana.rap-analyzer.local
- Username: admin
- Password: admin123 (change in values.yaml)

#### Prometheus Access
- URL: http://prometheus.rap-analyzer.local
- Metrics endpoint: /metrics

#### Available Dashboards
- Request Rate & Error Rate
- Response Time Percentiles
- Memory & CPU Usage
- Database Connections
- Custom Rap Analyzer Metrics

### Troubleshooting

#### Common Issues

1. **Pods stuck in Pending**
   ```bash
   kubectl describe pod <pod-name> -n rap-analyzer
   # Check resource availability and storage classes
   ```

2. **Database connection failed**
   ```bash
   kubectl logs deployment/rap-analyzer -n rap-analyzer
   kubectl exec -it deployment/postgres -- psql -U rap_analyzer -d rap_lyrics
   ```

3. **Ingress not working**
   ```bash
   kubectl get ingress -n rap-analyzer
   kubectl describe ingress rap-analyzer-ingress -n rap-analyzer
   ```

#### Logs Access

```bash
# API logs
kubectl logs deployment/rap-analyzer -f -n rap-analyzer

# Database logs  
kubectl logs deployment/postgres -f -n rap-analyzer

# Monitoring logs
kubectl logs deployment/prometheus -f -n rap-analyzer
kubectl logs deployment/grafana -f -n rap-analyzer
```

### Performance Tuning

#### Database Optimization

PostgreSQL is configured with optimized settings:
- Max connections: 200
- Shared buffers: 256MB
- Effective cache size: 1GB
- Connection pooling via application

#### API Optimization

- Multi-replica deployment (3-10 pods)
- Resource limits and requests
- Horizontal Pod Autoscaling
- Connection pooling to database

### Backup and Recovery

#### Database Backup

```bash
# Create backup job
kubectl create job --from=cronjob/postgres-backup postgres-backup-manual -n rap-analyzer

# Manual backup
kubectl exec deployment/postgres -n rap-analyzer -- \
  pg_dump -U rap_analyzer rap_lyrics > backup.sql
```

#### Configuration Backup

```bash
# Export Helm values
helm get values rap-analyzer -n rap-analyzer > current-values.yaml

# Export all K8s resources
kubectl get all -n rap-analyzer -o yaml > rap-analyzer-backup.yaml
```

### Uninstall

```bash
# Using Helm
helm uninstall rap-analyzer -n rap-analyzer
kubectl delete namespace rap-analyzer

# Manual cleanup
kubectl delete -f k8s/ --recursive
```

### Next Steps

- Phase 2: Advanced monitoring with Jaeger tracing
- Phase 3: Multi-region deployment
- Phase 4: GitOps integration with ArgoCD
- Security hardening with Pod Security Standards