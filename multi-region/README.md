# Multi-Region Deployment Guide

## Overview

This guide covers the deployment and management of the Rap Analyzer system across multiple geographical regions using Kubernetes and PostgreSQL streaming replication.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   US-East-1     │    │   US-West-2     │    │   EU-West-1     │
│   (Primary)     │    │   (Replica)     │    │   (Replica)     │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ PostgreSQL  │ │───▶│ │ PostgreSQL  │ │    │ │ PostgreSQL  │ │
│ │   Primary   │ │    │ │   Replica   │ │    │ │   Replica   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │        │        │    │        │        │
│ ┌─────────────┐ │    │        ▼        │    │        ▼        │
│ │ Rap Analyzer│ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ API (Write) │ │    │ │ Rap Analyzer│ │    │ │ Rap Analyzer│ │
│ └─────────────┘ │    │ │ API (Read)  │ │    │ │ API (Read)  │ │
│                 │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Infrastructure Requirements

1. **Kubernetes Clusters** (1.20+):
   - US-East-1 cluster (primary)
   - US-West-2 cluster (replica)
   - EU-West-1 cluster (replica)

2. **Network Configuration**:
   - Cross-region connectivity between clusters
   - PostgreSQL replication port (5432) accessible between regions
   - Load balancer configuration for regional traffic routing

3. **Tools**:
   - `kubectl` configured with contexts for each region
   - `helm` 3.0+
   - `argocd` CLI (optional)
   - PowerShell 5.1+ (for deployment script)

### Kubernetes Contexts

Ensure you have contexts configured for each region:

```bash
kubectl config get-contexts
```

Expected output:
```
CURRENT   NAME         CLUSTER       AUTHINFO      NAMESPACE
*         us-east-1    us-east-1     us-east-1     
          us-west-2    us-west-2     us-west-2     
          eu-west-1    eu-west-1     eu-west-1     
```

## Configuration Files

### Region-Specific Values

Each region has its own values file in `multi-region/clusters/`:

- `values-us-east-1.yaml` - Primary region (read/write)
- `values-us-west-2.yaml` - US West replica (read-only)
- `values-eu-west-1.yaml` - EU West replica (read-only, GDPR compliant)

### PostgreSQL Configuration

- `postgresql-primary.yaml` - Primary database configuration
- `postgresql-replica.yaml` - Replica database configuration  
- `postgresql-shared.yaml` - Shared resources (monitoring, backups)

## Deployment Process

### 1. Automated Deployment

Use the PowerShell deployment script for automated setup:

```powershell
# Deploy to all regions
.\multi-region\deploy-multi-region.ps1 -Action deploy

# Deploy to specific region
.\multi-region\deploy-multi-region.ps1 -Action deploy -Region us-east-1

# Dry run (preview changes)
.\multi-region\deploy-multi-region.ps1 -Action deploy -DryRun
```

### 2. Manual Deployment Steps

#### Step 1: Deploy Primary Region (US-East-1)

```bash
# Switch to primary cluster
kubectl config use-context us-east-1

# Deploy PostgreSQL primary
kubectl apply -f multi-region/postgresql/postgresql-shared.yaml
kubectl apply -f multi-region/postgresql/postgresql-primary.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgresql,role=primary -n rap-analyzer --timeout=300s

# Deploy application
helm upgrade --install rap-analyzer-us-east-1 ./helm/rap-analyzer \
  --namespace rap-analyzer \
  --create-namespace \
  --values helm/rap-analyzer/values.yaml \
  --values multi-region/clusters/values-us-east-1.yaml \
  --set global.region=us-east-1 \
  --set global.zone=primary \
  --wait --timeout=10m
```

#### Step 2: Deploy Replica Regions

**US-West-2:**

```bash
# Switch to west cluster
kubectl config use-context us-west-2

# Deploy PostgreSQL replica
kubectl apply -f multi-region/postgresql/postgresql-shared.yaml
kubectl apply -f multi-region/postgresql/postgresql-replica.yaml

# Deploy application
helm upgrade --install rap-analyzer-us-west-2 ./helm/rap-analyzer \
  --namespace rap-analyzer \
  --create-namespace \
  --values helm/rap-analyzer/values.yaml \
  --values multi-region/clusters/values-us-west-2.yaml \
  --set global.region=us-west-2 \
  --set global.zone=replica \
  --wait --timeout=10m
```

**EU-West-1:**

```bash
# Switch to EU cluster
kubectl config use-context eu-west-1

# Deploy PostgreSQL replica (with EU region label)
kubectl apply -f multi-region/postgresql/postgresql-shared.yaml
cat multi-region/postgresql/postgresql-replica.yaml | \
  sed 's/region: us-west-2/region: eu-west-1/' | \
  kubectl apply -f -

# Deploy application
helm upgrade --install rap-analyzer-eu-west-1 ./helm/rap-analyzer \
  --namespace rap-analyzer \
  --create-namespace \
  --values helm/rap-analyzer/values.yaml \
  --values multi-region/clusters/values-eu-west-1.yaml \
  --set global.region=eu-west-1 \
  --set global.zone=replica \
  --wait --timeout=10m
```

#### Step 3: Deploy ArgoCD ApplicationSet

```bash
# Switch back to primary cluster
kubectl config use-context us-east-1

# Deploy ApplicationSet for ongoing management
kubectl apply -f multi-region/clusters/multi-region-clusters.yaml
```

## Management & Operations

### Status Monitoring

Check deployment status across all regions:

```powershell
.\multi-region\deploy-multi-region.ps1 -Action status
```

Or manually check each region:

```bash
# Check PostgreSQL replication status
kubectl exec -it postgresql-primary-0 -n rap-analyzer -- \
  psql -U postgres -c "SELECT * FROM pg_stat_replication;"

# Check application health
kubectl get pods -n rap-analyzer -o wide
kubectl get svc -n rap-analyzer
```

### PostgreSQL Replication Monitoring

**Primary Database:**

```sql
-- Check replication slots
SELECT slot_name, active, restart_lsn FROM pg_replication_slots;

-- Check connected replicas
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn 
FROM pg_stat_replication;
```

**Replica Databases:**

```sql
-- Check replica status
SELECT pg_is_in_recovery();

-- Check replication lag
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()));
```

### Backup Operations

Automated backups run daily via CronJob:

```bash
# Check backup status
kubectl get cronjobs -n rap-analyzer

# View backup logs
kubectl logs -l job-name=postgresql-backup -n rap-analyzer
```

### Failover Procedures

#### Planned Failover

1. **Prepare for failover:**
   ```bash
   # Stop writes to primary
   kubectl scale deployment rap-analyzer --replicas=0 -n rap-analyzer
   
   # Ensure all replicas are caught up
   # Check replication lag on all replicas
   ```

2. **Promote replica to primary:**
   ```bash
   # Connect to target replica
   kubectl exec -it postgresql-replica-0 -n rap-analyzer -- \
     pg_ctl promote -D /var/lib/postgresql/data
   ```

3. **Update application configuration:**
   ```bash
   # Update Helm values to point to new primary
   helm upgrade rap-analyzer ./helm/rap-analyzer \
     --set database.host=new-primary-host \
     --reuse-values
   ```

#### Emergency Failover

Use the automated failover script:

```powershell
.\multi-region\deploy-multi-region.ps1 -Action failover -Force
```

## Troubleshooting

### Common Issues

#### PostgreSQL Replication Issues

**Problem**: Replica not connecting to primary

**Solution**:
```bash
# Check network connectivity
kubectl exec -it postgresql-replica-0 -n rap-analyzer -- \
  nc -zv postgresql-primary 5432

# Check replication user permissions
kubectl exec -it postgresql-primary-0 -n rap-analyzer -- \
  psql -U postgres -c "\du replicator"

# Restart replica
kubectl rollout restart statefulset postgresql-replica -n rap-analyzer
```

**Problem**: High replication lag

**Solution**:
```bash
# Check network latency between regions
# Increase wal_keep_segments on primary
# Consider using replication slots
```

#### Application Connection Issues

**Problem**: Application can't connect to database

**Solution**:
```bash
# Check service endpoints
kubectl get endpoints postgresql-primary -n rap-analyzer

# Verify connection string
kubectl describe configmap rap-analyzer-config -n rap-analyzer

# Check application logs
kubectl logs -l app=rap-analyzer -n rap-analyzer --tail=100
```

### Performance Optimization

#### Database Performance

```sql
-- Optimize PostgreSQL for replication
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET max_replication_slots = 10;
SELECT pg_reload_conf();
```

#### Application Performance

```yaml
# Optimize connection pooling in values.yaml
database:
  maxConnections: 20
  connectionPool:
    enabled: true
    maxSize: 10
    minSize: 2
```

## Security Considerations

### Network Security

- PostgreSQL replication traffic should use SSL/TLS
- Implement network policies for cross-region communication
- Use VPN or private networking for replication traffic

### Data Privacy (GDPR)

EU-West-1 cluster includes GDPR-specific configurations:

```yaml
# values-eu-west-1.yaml
gdpr:
  enabled: true
  dataRetention: "2 years"
  anonymization: true
  rightToForget: true
```

### Access Control

- Use separate RBAC policies per region
- Implement pod security policies
- Regular security audits and updates

## Monitoring & Alerting

### Metrics to Monitor

1. **Database Metrics**:
   - Replication lag
   - Connection count
   - Query performance
   - Disk usage

2. **Application Metrics**:
   - Response times per region
   - Error rates
   - Request volume
   - Resource utilization

3. **Infrastructure Metrics**:
   - Cluster health
   - Network latency between regions
   - Resource consumption

### Recommended Alerts

```yaml
# Example Prometheus alerts
- alert: PostgreSQLReplicationLag
  expr: pg_replication_lag_seconds > 300
  for: 5m
  annotations:
    summary: "PostgreSQL replication lag is high"

- alert: ApplicationDown
  expr: up{job="rap-analyzer"} == 0
  for: 2m
  annotations:
    summary: "Rap Analyzer application is down"
```

## Maintenance

### Regular Tasks

1. **Weekly**:
   - Review replication status
   - Check backup completion
   - Monitor resource usage

2. **Monthly**:
   - Update application images
   - Review and rotate secrets
   - Performance optimization review

3. **Quarterly**:
   - Disaster recovery testing
   - Security audit
   - Capacity planning review

### Upgrade Procedures

1. **Rolling Updates**:
   ```bash
   # Update application in each region
   helm upgrade rap-analyzer-us-east-1 ./helm/rap-analyzer \
     --set image.tag=new-version \
     --reuse-values
   ```

2. **Database Updates**:
   - Test on replica first
   - Coordinate with maintenance windows
   - Have rollback plan ready

## Cost Optimization

### Regional Cost Considerations

- **US-East-1**: Primary region, higher compute needs
- **US-West-2**: Replica region, optimized for read workloads  
- **EU-West-1**: Compliance region, balanced configuration

### Resource Optimization

```yaml
# Example resource limits per region
resources:
  primary:
    requests: { cpu: "2", memory: "4Gi" }
    limits: { cpu: "4", memory: "8Gi" }
  replica:
    requests: { cpu: "1", memory: "2Gi" }
    limits: { cpu: "2", memory: "4Gi" }
```

## Support & Documentation

For additional support:

1. Check application logs: `kubectl logs -l app=rap-analyzer -n rap-analyzer`
2. Review PostgreSQL logs: `kubectl logs postgresql-primary-0 -n rap-analyzer`
3. Monitor system metrics in Grafana dashboard
4. Consult troubleshooting section above

---

**Last Updated**: January 2025  
**Version**: 2.0.0 (Multi-Region Deployment)  
**Author**: AI Engineering Team